#redis + caching logic

import redis
import hashlib
import pickle

r = redis.Redis(host='localhost', port=6379, db=0)

def cache_prediction(image_bytes):
    key = hashlib.md5(image_bytes).hexdigest()
    cached = r.get(key)
    if cached:
        return pickle.loads(cached)
    return None

def store_prediction(image_bytes, result):
    key = hashlib.md5(image_bytes).hexdigest()
    r.set(key, pickle.dumps(result), ex=3600)  # expire in 1 hour
