from pymongo import MongoClient
import pandas as pd
from datetime import datetime, timedelta

class MongoDBHandler:
    """
    Class for managing the connection and handling of a MongoDB database.
    """
    def __init__(self, uri: str, db_name: str):
        """
        Initializes the connection to the MongoDB database.
        If the database does not exist, it will be created.
        
        :param uri: MongoDB connection URI
        :param db_name: Name of the database to be used
        """
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        
        # Ensure database creation by accessing a dummy collection
        self.db.create_collection("init_collection", capped=True, size=4096) if "init_collection" not in self.db.list_collection_names() else None
        
        print(f"Connected to the database: {db_name}")

    def close_connection(self):
        """
        Closes the connection to the database.
        """
        self.client.close()
        print("MongoDB connection closed.")

    def insert_record(self, table_name: str, record: dict):
        """
        Inserts a single record into the specified collection.
        
        :param table_name: Name of the collection where the document will be inserted
        :param record: Dictionary representing the document to be inserted
        """
        collection = self.db[table_name]
        collection.insert_one(record)
        print(f"Record inserted into '{table_name}' collection.")

    def fetch_all(self, table_name: str):
        """
        Retrieves all documents from the specified collection and returns them as a Pandas DataFrame.
        
        :param table_name: Name of the collection to fetch data from
        :return: Pandas DataFrame containing all documents in the collection
        """
        collection = self.db[table_name]
        data = list(collection.find())
        return pd.DataFrame(data)


    def create_table_from_csv(self, file_path: str, table_name: str, indexes: list):
        """
        Creates a table (collection) in the database from a CSV file.
        
        :param file_path: Path to the CSV file
        :param table_name: Name of the target collection
        :param indexes: List of fields to be used as indexes
        """
        df = pd.read_csv(file_path)
        collection = self.db[table_name]
        
        # Convert DataFrame to dictionary and insert into MongoDB
        data = df.to_dict(orient='records')
        collection.insert_many(data)
        
        # Create indexes if specified
        for index in indexes:
            collection.create_index([(index, 1)])
        
        print(f"Table '{table_name}' created with {len(df)} records.")

    def update_one(self, table_name: str, primary_key: str, field_name: str, field_value):
        """
        Atomically updates a specific field in a document identified by a unique primary key,
        ensuring that another worker did not update it in the meantime.
    
        :param table_name: Name of the collection where the document is located
        :param primary_key: The unique identifier (primary key) of the document
        :param field_name: The field to be updated
        :param field_value: The new value to set in the specified field
        """
        collection = self.db[table_name]
    
        result = collection.update_one(
            {"_id": primary_key, field_name: {"$ne": field_value}},  # Apenas atualiza se o valor for diferente
            {"$set": {field_name: field_value, "last_modified": datetime.utcnow()}}
        )
    
        if result.modified_count > 0:
            print(f"Document with ID '{primary_key}' updated successfully.")
        else:
            print(f"No update performed for ID '{primary_key}' (already updated or missing).")

            
    def get_next_record(self, table_name: str, control_field: str, select_value: str, update_value: str, worker_id: str, timeout_minutes: int = 10, quantity: int = 1):
        """
        Retrieves the next available records from the specified collection where control_field == select_value
        and marks them as processing by the given worker, ensuring atomic updates.
    
        :param table_name: Name of the collection to retrieve the records from
        :param control_field: The field that controls processing
        :param select_value: The value to search for in control_field
        :param update_value: The value to set in control_field
        :param worker_id: Unique identifier for the worker processing the record
        :param timeout_minutes: Time (in minutes) after which a record is considered unprocessed
        :param quantity: Number of records to retrieve and update (default: 1)
        :return: Pandas DataFrame containing the retrieved documents or None if no records are available
        """
        collection = self.db[table_name]
        now = datetime.utcnow()
        timeout_threshold = now - timedelta(minutes=timeout_minutes)
    
        documents = []
    
        for _ in range(quantity):
            document = collection.find_one_and_update(
                {
                    "$or": [
                        {control_field: select_value},  # Registros disponíveis
                        {  # Registros que estão travados por outro worker, mas já passaram do timeout
                            control_field: update_value,
                            "processing_timestamp": {"$lt": timeout_threshold}
                        }
                    ]
                },
                {
                    "$set": {
                        control_field: update_value,
                        "processing_by": worker_id,
                        "processing_timestamp": now
                    }
                },
                return_document=True
            )
    
            if document:
                documents.append(document)
            else:
                break  # Se não houver mais registros disponíveis, parar
    
        return pd.DataFrame(documents) if documents else None




    def add_field_to_collection(self, table_name: str, field_name: str, field_type: type, default_value, index_field: bool = False):
        """
        Adds a new field to all documents in a collection, ensuring it has the specified type.
        Optionally creates an index for the field.
    
        :param table_name: Name of the collection to modify
        :param field_name: The name of the new field to add
        :param field_type: The expected data type (str, int, float, bool, etc.)
        :param default_value: The default value to set for this field
        :param index_field: If True, an index will be created for this field (default: False)
        """
        collection = self.db[table_name]
    
        # Ensure the default value is of the correct type
        try:
            typed_default_value = field_type(default_value)
        except ValueError:
            raise ValueError(f"Default value '{default_value}' cannot be converted to type {field_type.__name__}")
    
        # Update all documents by adding the new field if it does not exist
        result = collection.update_many(
            {field_name: {"$exists": False}},  # Only update if the field does not exist
            {"$set": {field_name: typed_default_value}}
        )
    
        print(f"Field '{field_name}' added to {result.modified_count} documents in '{table_name}' collection.")
    
        # Create an index if index_field is True
        if index_field:
            collection.create_index([(field_name, 1)])
            print(f"Index created for field '{field_name}' in collection '{table_name}'.")

    def create_table_from_dataframe(
        self, 
        df: pd.DataFrame, 
        table_name: str, 
        indexes: list = None, 
        reference_table: str = None, 
        key_field: list | str = None, 
        update_field: str = None, 
        update_value: str = None,
        sort_by: str = None, 
        sort_order: int = -1
    ):
        """
        Inserts a Pandas DataFrame into a MongoDB collection.
        Optionally, updates a field in a reference collection if the key_field(s) exist.
        Also allows sorting the data before insertion.
    
        :param df: Pandas DataFrame containing the data to insert
        :param table_name: Name of the target collection
        :param indexes: List of fields to be used as indexes (default: None)
        :param reference_table: Name of the reference collection to update (optional)
        :param key_field: Field(s) used as a key for matching records between tables (single field or list of fields) (optional)
        :param update_field: The field in the reference table to update (optional)
        :param update_value: The new value to set in the update_field (optional)
        :param sort_by: Field to sort the DataFrame before inserting into MongoDB (optional)
        :param sort_order: Sorting order, 1 for ascending, -1 for descending (default: -1, descending)
        """
        if df.empty:
            print(f"DataFrame is empty. No data inserted into '{table_name}'.")
            return
        
        collection = self.db[table_name]
    
        # Apply sorting if a sorting field is provided
        if sort_by and sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending=(sort_order == 1))
    
        # Convert DataFrame to dictionary format
        data = df.to_dict(orient='records')
    
        # Insert records
        collection.insert_many(data)
        
        # Create indexes if specified
        if indexes:
            for index in indexes:
                collection.create_index([(index, 1)])
            print(f"Indexes created for fields {indexes} in collection '{table_name}'.")
    
        print(f"Table '{table_name}' updated with {len(data)} new records.")
    
        # If reference_table parameters are provided, update the reference collection
        if reference_table and key_field and update_field and update_value:
            reference_collection = self.db[reference_table]
    
            if isinstance(key_field, str):
                key_field = [key_field]
    
            inserted_keys = [{field: record[field] for field in key_field if field in record} for record in data]
            inserted_keys = [keys for keys in inserted_keys if keys]  
    
            if inserted_keys:
                result = reference_collection.update_many(
                    {"$or": inserted_keys},
                    {"$set": {update_field: update_value}}
                )
                print(f"Updated {result.modified_count} records in '{reference_table}' where {key_field} matches new records.")

    def get_first_record(self, table_name: str, sort_by: str = None, sort_order: int = -1):
        """
        Retrieves the first record from a collection, optionally applying sorting.
    
        :param table_name: Name of the collection to retrieve the record from
        :param sort_by: Field to sort the records before retrieving the first one (optional)
        :param sort_order: Sorting order, 1 for ascending (smallest first), -1 for descending (largest first) (default: -1)
        :return: The first record as a Pandas DataFrame or None if no records exist
        """
        collection = self.db[table_name]
    
        # Aplicar ordenação, se fornecida
        if sort_by:
            document = collection.find_one({}, sort=[(sort_by, sort_order)])
        else:
            document = collection.find_one({})  # Retorna o primeiro documento sem ordenação
    
        return pd.DataFrame([document]) if document else None

    def set_metadata(self, key: str, value):
        """
        Stores or updates a metadata key-value pair in the 'metadata' collection.
    
        :param key: The unique key identifying the configuration setting.
        :param value: The value to store under the given key.
        """
        collection = self.db["metadata"]
    
        result = collection.update_one(
            {"key": key},   # Filtra pela chave
            {"$set": {"value": value}},  # Atualiza o valor
            upsert=True  # Se a chave não existir, cria um novo documento
        )
    
        if result.upserted_id:
            print(f"New metadata '{key}' created.")
        else:
            print(f"Metadata '{key}' updated.")
    
    def get_metadata(self, key: str):
        """
        Retrieves the value of a metadata key from the 'metadata' collection.
    
        :param key: The unique key identifying the configuration setting.
        :return: The stored value or None if the key is not found.
        """
        collection = self.db["metadata"]
    
        document = collection.find_one({"key": key})  # Busca o documento com a chave específica
    
        return document["value"] if document else None  # Retorna o valor se encontrado, caso contrário, retorna None

    def update_all(self, table_name: str, field_name: str, field_value):
        """
        Updates a specific field for all documents in a given collection.
    
        :param table_name: Name of the collection where documents will be updated.
        :param field_name: The field to be updated.
        :param field_value: The new value to set for the specified field.
        """
        collection = self.db[table_name]
    
        result = collection.update_many({}, {"$set": {field_name: field_value}})
    
        print(f"Updated {result.modified_count} records in '{table_name}' setting '{field_name}' to '{field_value}'.")

