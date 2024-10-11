import math
from datetime import datetime
#2h

def get_user_data_():
    name = input("Enter your name: ")
    day_of_birth = input("Enter your day of birth: ")
    month_of_birth = input("Enter your month of birth: ")
    year_of_birth = input("Enter your year of birth: ")
    return name, day_of_birth, month_of_birth, year_of_birth

def is_year_leap(year):
    if year%4==0 and year%100!=0 :
        return True
    elif year%4==0 and year%100==0 and year%400!=0 :
        return False

def calculate_month_days(month):
    month_days=0
    for i in range (month):
        if month in [1, 3, 5, 7, 8, 10, 12]:
            month_days += 31
        elif month in [4, 6, 9, 11]:
            month_days += 30
        else:
            month_days += 28

    return month_days

def calculate_day_age(day_of_birth, month_of_birth, year_of_birth):
    year_today = datetime.now().year
    month_today = datetime.now().month
    day_today = datetime.now().day
    years = year_today - year_of_birth

    if month_today >= month_of_birth:
        months = month_today - month_of_birth
    else:
        months = 12 - month_of_birth + month_today
        years -= 1
    if day_today > day_of_birth:
        days = day_today - day_of_birth
    else:
        days = 30 - day_of_birth + day_today
        months -= 1

    days_age=calculate_month_days(months) + days

    for i in range(year_today-year_of_birth):
        if not year_today-year_of_birth<=1 :
            days_age += 365
            if is_year_leap(i):
                days_age += 1

    return days_age

def calculate_emotional_wave(years,months,days):
    emotional_wave = math.sin((2*math.pi/33)*calculate_day_age(days,months,years))
    return  emotional_wave

def calculate_intellectual_wave(years, months, days):
    intellectual_wave = math.sin((2*math.pi/28)*calculate_day_age(days,months,years))
    return  intellectual_wave

def calculate_physical_wave(years,months,days):
    physical_wave = math.sin((2*math.pi/23)*calculate_day_age(days,months,years))
    return  physical_wave

def display_user_data(name, years, months, days):
    day_age = calculate_day_age(days, months, years)
    emotional_wave = calculate_emotional_wave(years, months, days)
    physical_wave = calculate_physical_wave(years, months, days)
    intellectual_wave = calculate_intellectual_wave(years, months, days)

    print(f"\n Hello {name}! Your day age is {day_age}" )
    print(f"\n Your emotional wave measure is {emotional_wave} ,physical is {physical_wave} and intellectual is {intellectual_wave} ")

    if emotional_wave < -0.5 or physical_wave < -0.5 or intellectual_wave < -0.5 :
        print("Don't worry be happy ! :)")
        if calculate_physical_wave(years, months, days+1) > physical_wave and calculate_emotional_wave(years, months, days+1)> emotional_wave and calculate_intellectual_wave(years, months, days+1) > intellectual_wave :
            print("Tomorrow will be better")
    elif emotional_wave > 0.5 or physical_wave > 0.5 or intellectual_wave > 0.5 :
        print("Have a great day !")

def main():
    name, day_of_birth, month_of_birth, year_of_birth = get_user_data_()
    display_user_data(name,int(year_of_birth),int(month_of_birth),int(day_of_birth))

if __name__ == "__main__":
    main()