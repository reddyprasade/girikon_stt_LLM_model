import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
# __DB_LOCAL = os.getenv("DB_LOCAL")
# __user_name = os.getenv("DB_USER")
# __password = os.getenv("DB_PASS")
# __host = os.getenv("DB_HOST")
# __port = os.getenv("DB_PORT")
# __db_name = os.getenv("DB_NAME")
# __mechanism = os.getenv("DB_MECHANISM")
# __mongo_uri = f"mongodb+srv://{__user_name}:{__password}@{__host}:{__port}/?authSource={__db_name}&authMechanism={__mechanism}&retryWrites=true&loadBalanced=false&serverSelectionTimeoutMS=5000&connectTimeoutMS=10000"

# try:
#     if __DB_LOCAL == "True" or __DB_LOCAL == "true":
#         __client = MongoClient("mongodb://localhost:27017")
#     else:
#         __client = MongoClient(
#             host=f"{__host}:{__port}",
#             username=__user_name,
#             password=__password,
#             authSource=__db_name,
#             authMechanism=__mechanism,
#             retryWrites=True,
#             serverSelectionTimeoutMS=10000,
#             connectTimeoutMS=20000
#         )

# except Exception as e:
#     print(e)
#     print("Error in connecting to mongo db")
#     __client = None



# def getCurrentMongoUri():
#     return __mongo_uri


# def getCurrentDBName():
#     return __db_name


# def getMongoClient(database_name=None):
#     if database_name is None:
#         return __client[__db_name]
#     else:
#         return __client[database_name]


# def getMongoClientClose(client):
#     try:
#         client.close()
#     except Exception as e:
#         print(e)
#         return False
#     return True


# db = getMongoClient()


def voice_transcribe_grammatical_correction_emotion(data):
    """
    Stores the data in the specified collection.

    Args:
        data (dict): The data to be stored.

    Returns:
        bool: True if the data is stored successfully, False otherwise.
    """
    try:
        from pymongo import MongoClient
        import os 
        from dotenv import load_dotenv
        load_dotenv()
        user_name = os.getenv("DB_USER")
        password = os.getenv("DB_PASS")
        db_name = os.getenv("DB_NAME")
        host = os.getenv("DB_HOST")
        port = os.getenv("DB_PORT")
        mechanism = os.getenv("DB_MECHANISM")
        collection_name = 'voice_transcribe_grammatical_correction_emotion'
        mongo_uri = f"mongodb://{user_name}:{password}@{host}:{port}/?authSource={db_name}&authMechanism={mechanism}&retryWrites=true&loadBalanced=false&serverSelectionTimeoutMS=5000&connectTimeoutMS=10000"
        client = MongoClient(mongo_uri)
        collection = client[db_name][collection_name]
        result = collection.insert_one(data)
        return result.acknowledged
    except Exception as e:
        print(e)
        return False