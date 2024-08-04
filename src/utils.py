import logging

def setup_logging():
    logging.basicConfig(filename='logs/app.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def log_error(e):
    logging.error(f"Error: {e}")