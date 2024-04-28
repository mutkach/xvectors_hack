import json
import time

import numpy as np
import pandas as pd
import redis
import requests
from redis.commands.search.field import (
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer


client = redis.Redis(host="localhost", port=6379, decode_responses=True)


print(json.dumps(bikes[0], indent=2))
 
