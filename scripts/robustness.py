import time
from functools import wraps

def retry_with_backoff(max_retries=3, initial_delay=1, backoff_factor=2, exceptions=(Exception,)):
    """
    Decorator to retry a function call with exponential backoff.
    
    Args:
        max_retries (int): Maximum number of retries.
        initial_delay (float): Initial delay in seconds.
        backoff_factor (float): Multiplier for delay after each failure.
        exceptions (tuple): Tuple of exceptions to catch and retry on.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        print(f"⚠️ Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        print(f"❌ All {max_retries} attempts failed.")
            
            if last_exception:
                raise last_exception
        return wrapper
    return decorator
