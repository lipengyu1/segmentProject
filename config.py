import secrets

class Config:
    secret_key = secrets.token_hex(16)
    DEBUG = False
    SECRET_KEY = secret_key  # 替换为安全的随机字符串，例如 secrets.token_hex(16)
    UPLOAD_FOLDER = 'datasets/'
    PROCESSED_FOLDER = 'static/processed'