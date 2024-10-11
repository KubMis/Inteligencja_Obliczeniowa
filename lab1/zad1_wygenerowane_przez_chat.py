import math
from datetime import datetime, timedelta


# Funkcja do obliczania biorytmów
def oblicz_biorytm(dni_zycia, cykl):
    return math.sin(2 * math.pi * dni_zycia / cykl)


# Funkcja do obliczania dni życia na podstawie daty urodzenia
def oblicz_dni_zycia(data_urodzenia):
    dzis = datetime.now()
    roznica = dzis - data_urodzenia
    return roznica.days


# Funkcja do wyświetlania komunikatów w zależności od wyniku
def analizuj_wynik(wynik, nazwa_cyklu, imie):
    if wynik > 0.5:
        print(f"Gratulacje, {imie}! Twój {nazwa_cyklu} biorytm jest wysoki: {wynik:.2f}.")
    elif wynik < -0.5:
        print(f"Nie martw się, {imie}, Twój {nazwa_cyklu} biorytm jest obecnie niski: {wynik:.2f}.")
        return True
    else:
        print(f"Twój {nazwa_cyklu} biorytm jest w normie: {wynik:.2f}.")
    return False


# Funkcja do obliczania biorytmów na podstawie daty
def oblicz_i_analizuj_biorytmy(data_urodzenia, imie):
    dni_zycia = oblicz_dni_zycia(data_urodzenia)
    # Cykl fizyczny, emocjonalny i intelektualny
    cykle = {'fizyczny': 23, 'emocjonalny': 28, 'intelektualny': 33}

    print(f"\nBiorytmy dla {imie} na dzień dzisiejszy:\n")
    wyniki = {}

    for nazwa_cyklu, cykl in cykle.items():
        wynik = oblicz_biorytm(dni_zycia, cykl)
        wyniki[nazwa_cyklu] = wynik
        trzeba_sprawdzic_jutro = analizuj_wynik(wynik, nazwa_cyklu, imie)

        # Jeśli wynik jest niski, sprawdź jutro
        if trzeba_sprawdzic_jutro:
            sprawdz_jutro(data_urodzenia, nazwa_cyklu, cykl, imie)


def sprawdz_jutro(data_urodzenia, nazwa_cyklu, cykl, imie):
    dni_zycia_jutro = oblicz_dni_zycia(data_urodzenia) + 1
    wynik_jutro = oblicz_biorytm(dni_zycia_jutro, cykl)

    print(f"\nSprawdzam biorytm {nazwa_cyklu} na jutro...")
    if wynik_jutro > wynik_jutro:
        print(f"Dobra wiadomość, {imie}! Jutro Twój {nazwa_cyklu} biorytm będzie lepszy: {wynik_jutro:.2f}.\n")
    else:
        print(f"Jutro Twój {nazwa_cyklu} biorytm nieznacznie się zmieni: {wynik_jutro:.2f}.\n")


# Główna funkcja programu
def main():
    # Pobieranie danych od użytkownika
    imie = input("Podaj swoje imię: ")
    rok = int(input("Podaj rok urodzenia (np. 1990): "))
    miesiac = int(input("Podaj miesiąc urodzenia (np. 7): "))
    dzien = int(input("Podaj dzień urodzenia (np. 15): "))

    # Tworzenie daty urodzenia
    data_urodzenia = datetime(rok, miesiac, dzien)

    # Obliczanie i analiza biorytmów
    oblicz_i_analizuj_biorytmy(data_urodzenia, imie)


# Uruchomienie programu
if __name__ == "__main__":
    main()
