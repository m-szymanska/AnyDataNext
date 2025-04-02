# Status rozwoju AnyDataset

> Data ostatniej aktualizacji: 1 kwietnia 2025

## Obecny stan projektu

Projekt jest w fazie beta. Zaimplementowana została podstawowa struktura aplikacji z interfejsem webowym i możliwością konwersji różnych formatów danych. Aplikacja działa stabilnie, a kluczowe funkcje z planowanej wersji 1.0 są sukcesywnie dodawane.

## Ukończone elementy

- [x] Podstawowy szkielet projektu Flask
- [x] Główny interfejs aplikacji webowej
- [x] Obsługa konwersji plików TXT
- [x] Obsługa konwersji plików MD
- [x] Obsługa konwersji plików CSV/TSV
- [x] Obsługa konwersji plików JSON/JSONL
- [x] Obsługa konwersji plików YAML
- [x] Obsługa konwersji plików PDF
- [x] Obsługa konwersji plików DOCX
- [x] Szablony HTML dla interfejsu użytkownika
- [x] System logowania operacji
- [x] Obsługa przesyłania plików
- [x] Zarządzanie postępem konwersji
- [x] Parsowanie danych z różnych formatów
- [x] Generowanie raportów z rozumowaniem
- [x] Eksport danych do formatów strukturalnych

## Do zaimplementowania w następnym sprincie

- [ ] Obsługa plików audio (WAV, MP3, M4A)
- [ ] Obsługa plików SQL/DB
- [ ] Interfejs API REST
- [ ] Rozszerzona dokumentacja użytkownika
- [ ] Anonimizacja danych wrażliwych
- [ ] Automatyczne tagowanie danych

## Formaty obsługiwane i planowane

### Obsługiwane obecnie
- [x] TXT - z chunkingiem akapitów
- [x] MD - z obsługą formatowania i struktury
- [x] CSV/TSV - z obsługą tabelaryczną
- [x] JSON/JSONL - z obsługą struktur zagnieżdżonych
- [x] YAML - z parsowaniem strukturalnym
- [x] PDF - z ekstrakcją tekstu
- [x] DOCX - z zachowaniem formatowania

### Planowane wsparcie
- [ ] WAV/MP3/M4A - z transkrypcją audio
- [ ] SQL/DB - z importem i eksportem danych
- [ ] XML - z parsowaniem strukturalnym

## Strategie parsowania i chunkowania

### 1. Pliki proste (TXT, MD)
- Dzielenie tekstu po podwójnych newline (`\n\n`) na akapity
- Limitowanie rozmiaru akapitów do 2000 znaków
- Inteligentne dzielenie dłuższych fragmentów przy znakach przestankowych
- Zachowanie metadanych o numerze akapitu i przedziale znaków

### 2. Pliki tabelaryczne (CSV, TSV)
- Obsługa rekordów pojedynczych (jeden wiersz = jeden rekord)
- Obsługa rekordów wielokrotnych (łączenie powiązanych kolumn)
- Spłaszczanie złożonych struktur do formatu tekstowego
- Chunking długich tekstów w komórkach powyżej 2000 znaków

### 3. Pliki zagnieżdżone (JSON, JSONL, YAML, XML)
- Ekstrakcja ważnych elementów strukturalnych
- Tworzenie osobnych chunków dla każdej sekcji/elementu
- Podział długich treści na mniejsze fragmenty
- Zachowanie struktury i relacji między elementami

### 4. Dokumenty z formatowaniem (DOCX, PDF)
- Ekstrakcja tekstu z zachowaniem struktury dokumentu
- Wykrywanie nagłówków i sekcji
- Dzielenie długich akapitów
- Obsługa dokumentów wielostronicowych

## Znane problemy i ograniczenia

1. Konwersja dużych plików PDF może być wolna i pamięciochłonna
2. Brak pełnej obsługi plików audio i SQL
3. Ograniczona obsługa plików wielojęzycznych
4. Trudności z ekstrakcją tabel z PDF
5. Niekompletna obsługa formatowania w DOCX

## Priorytety na następny sprint

1. **Wysoki priorytet**:
   - [ ] Implementacja obsługi plików SQL/DB
   - [ ] Dodanie interfejsu API REST
   - [ ] Anonimizacja danych wrażliwych

2. **Średni priorytet**:
   - [ ] Obsługa plików audio (WAV/MP3/M4A)
   - [ ] Optymalizacja przetwarzania dużych plików PDF
   - [ ] Automatyczne tagowanie danych
   - [ ] Normalizacja pól (standaryzacja nazw)

3. **Niski priorytet**:
   - [ ] Spłaszczenie struktur JSON
   - [ ] Pełna dokumentacja użytkownika
   - [ ] Testy jednostkowe i integracyjne
   - [ ] Dodawanie meta-danych (source, label, date)

## Planowane transformacje

- [ ] Spłaszczenie struktur JSON
- [ ] Normalizacja pól (standaryzacja nazw)
- [ ] Automatyczne tagowanie danych
- [ ] Chunking tekstu na małe części
- [ ] Anonimizacja danych wrażliwych
- [ ] Dodawanie meta-danych (source, label, date)

## Notatki techniczne

- Projekt wymaga Python 3.8 lub nowszego
- Aplikacja używa FastAPI jako framework webowy
- Zaimplementowane jest logowanie operacji
- Możliwość przetwarzania równoległego dla operacji wymagających dużej mocy obliczeniowej

## Użyte narzędzia i biblioteki

- Python 3.8+
- FastAPI
- Uvicorn
- PyPDF2/pdfminer
- python-docx
- pandas
- PyYAML
- pytest (testy)
- librosa/pydub (planowane dla audio)

## Środowisko testowe

- macOS, Linux, Windows
- Python 3.8+
- Przeglądarki: Chrome, Firefox, Safari

---

*Uwaga: Aby uruchomić projekt, należy wykonać polecenie `python app/app.py` w katalogu głównym.*