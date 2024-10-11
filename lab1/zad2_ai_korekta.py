import math
import random
import matplotlib.pyplot as plt
import numpy as np

# Constants
INITIAL_VELOCITY = 50  # m/s
HEIGHT = 100  # meters
G = 9.81  # gravity constant


# Function to calculate the distance based on angle
def calculate_distance(angle):
    angle = math.radians(angle)
    sin_angle = math.sin(angle)
    cos_angle = math.cos(angle)
    term = (INITIAL_VELOCITY * sin_angle) + math.sqrt((INITIAL_VELOCITY ** 2 * sin_angle ** 2) + (2 * G * HEIGHT))
    return term * ((INITIAL_VELOCITY * cos_angle) / G)


# Function to calculate projectile trajectory for given angle and x-values
def calculate_trajectory(angle, x):
    angle = math.radians(angle)
    sin_angle = math.sin(angle)
    cos_angle = math.cos(angle)
    return (-(G / (2 * INITIAL_VELOCITY ** 2 * cos_angle ** 2)) * x ** 2) + ((sin_angle / cos_angle) * x) + HEIGHT


# Function to plot trajectory and target
def draw_plot(angle, distance, target_value):
    x_values = np.linspace(0, distance, 500)
    y_values = calculate_trajectory(angle, x_values)

    plt.plot(x_values, y_values, label="Projectile Trajectory")
    plt.axvline(x=target_value, color='r', linestyle='--', label=f'Target at {target_value}m')
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.xlabel("Distance (m)")
    plt.ylabel("Height (m)")
    plt.title(f"Projectile Trajectory at {angle}°")
    plt.legend()
    plt.grid(True)
    plt.savefig("trajectory_plot.png")
    plt.show()


# Function to generate random target distance
def get_target_value():
    return random.randint(50, 340)


# Main game function
def main():
    target_value = get_target_value()
    print(f"Target is {target_value} meters away.")
    attempts = 0

    while True:
        try:
            # Increment attempt count
            attempts += 1

            # Get user input for angle
            user_angle = float(input("Enter the shooting angle (in degrees): "))

            # Calculate where the projectile will land
            shot_distance = calculate_distance(user_angle)

            # Check if the shot is within 5 meters of the target
            if target_value - 5 <= shot_distance <= target_value + 5:
                print(f"Congratulations! You hit the target in {attempts} attempt(s).")
                draw_plot(user_angle, shot_distance, target_value)
                break
            else:
                # Provide feedback on how far the shot was
                difference = abs(target_value - shot_distance)
                print(
                    f"Your shot landed {shot_distance:.2f} meters away. You missed by {difference:.2f} meters. Try again!\n")

        except ValueError:
            # Handle invalid input
            print("Invalid input. Please enter a valid angle in degrees.\n")


if __name__ == "__main__":
    main()
