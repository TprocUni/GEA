import time
import random

def precise_seed_random():
    # Use performance counter for a more precise time measurement
    precise_time = time.perf_counter()
    random.seed(int(precise_time * 1_000_000))  # Convert to microseconds and seed

# Example usage
precise_seed_random()
random_number = random.randint(1, 100)
print(random_number)

# Repeat for a new random number
precise_seed_random()
another_random_number = random.randint(1, 100)
print(another_random_number)
