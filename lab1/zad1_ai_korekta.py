import math
from datetime import datetime

def get_user_data():
    """Pobiera dane użytkownika: imię oraz datę urodzenia."""
    name = input("Enter your name: ")
    day_of_birth = int(input("Enter your day of birth: "))
    month_of_birth = int(input("Enter your month of birth: "))
    year_of_birth = int(input("Enter your year of birth: "))
    return name, day_of_birth, month_of_birth, year_of_birth

def is_year_leap(year):
    """Sprawdza, czy rok jest przestępny."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def calculate_month_days(month, year):
    """Zwraca liczbę dni w danym miesiącu, uwzględniając rok przestępny."""
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    elif month in [4, 6, 9, 11]:
        return 30
    elif month == 2:
        return 29 if is_year_leap(year) else 28
    return 0

def calculate_day_age(day_of_birth, month_of_birth, year_of_birth):
    """Oblicza całkowitą liczbę dni życia."""
    today = datetime.now()
    birth_date = datetime(year_of_birth, month_of_birth, day_of_birth)
    delta = today - birth_date
    return delta.days

def calculate_emotional_wave(day_age):
    """Oblicza wartość fali emocjonalnej na podstawie liczby dni życia."""
    return math.sin((2 * math.pi / 33) * day_age)

def calculate_intellectual_wave(day_age):
    """Oblicza wartość fali intelektualnej na podstawie liczby dni życia."""
    return math.sin((2 * math.pi / 28) * day_age)

def calculate_physical_wave(day_age):
    """Oblicza wartość fali fizycznej na podstawie liczby dni życia."""
    return math.sin((2 * math.pi / 23) * day_age)

def display_user_data(name, day_age):
    """Wyświetla dane użytkownika oraz wyniki obliczeń fal biorytmicznych."""
    emotional_wave = calculate_emotional_wave(day_age)
    physical_wave = calculate_physical_wave(day_age)
    intellectual_wave = calculate_intellectual_wave(day_age)

    print(f"\nHello {name}! Your age in days is {day_age} days.")
    print(f"Your emotional wave is {emotional_wave:.2f}, physical wave is {physical_wave:.2f}, and intellectual wave is {intellectual_wave:.2f}.")

    if emotional_wave < -0.5 or physical_wave < -0.5 or intellectual_wave < -0.5:
        print("Don't worry, be happy! :)")
        if (calculate_physical_wave(day_age + 1) > physical_wave and
            calculate_emotional_wave(day_age + 1) > emotional_wave and
            calculate_intellectual_wave(day_age + 1) > intellectual_wave):
            print("Tomorrow will be better!")
    elif emotional_wave > 0.5 or physical_wave > 0.5 or intellectual_wave > 0.5:
        print("Have a great day!")

def main():
    """Główna funkcja programu."""
    name, day_of_birth, month_of_birth, year_of_birth = get_user_data()
    day_age = calculate_day_age(day_of_birth, month_of_birth, year_of_birth)
    display_user_data(name, day_age)

if __name__ == "__main__":
    main()
