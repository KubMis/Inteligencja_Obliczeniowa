import math
import random
import matplotlib.pyplot as plt
import numpy as np

INITIAL_VELOCITY = 50
HEIGHT = 100
G = 9.81

def calculate_distance(angle):
    angle=math.radians(angle)
    return (INITIAL_VELOCITY*math.sin(angle) + math.sqrt( INITIAL_VELOCITY**2 * math.sin(angle)**2 + (2*G*HEIGHT))) * ( (INITIAL_VELOCITY*math.cos(angle))/G)

def calculate_trajectory(angle, x):
    angle = math.radians(angle)
    sin=math.sin(angle)
    cos=math.cos(angle)
    return (-(G/(2*(INITIAL_VELOCITY**2 * cos**2)))*x**2) + ((sin/cos)*x) + HEIGHT

def draw_plot(angle,length):
    x_values=np.linspace(0,length,430)
    y_values=calculate_trajectory(angle, x_values)
    plt.plot(x_values,y_values, label="Trajektoria pocisku")
    plt.ylim(bottom=0,top = 300)
    plt.xlabel("Odległość x [m]")
    plt.ylabel("Wysokość y [m]")
    plt.savefig("trajektoria.png")

def get_target_value():
    return random.randrange(50, 340)

def main():
    target_value = get_target_value()
    print(f"Target is {target_value} meters away")
    attempts = 0
    while True :
        attempts+=1
        user_angle = input("Hit enemy target by shooting at angle \n")
        user_angle = int(user_angle)
        user_attempt = calculate_distance(int(user_angle))
        if target_value - 5 <= int(user_attempt) and int(user_attempt) <= target_value + 5 :
            print(f"Target hit congratulations! at attempt {attempts} \n")
            draw_plot(user_angle,target_value)
            break
        else :
            print(f"try again your projectile land {user_attempt} meters away \n")

if __name__ == "__main__":
    main()
