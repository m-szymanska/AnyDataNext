# AnyDataset - Wizja rozwoju

## Podsumowanie kluczowych funkcjonalności i przyszłych kierunków

### 1. Nowy interfejs Process File

- **Etapowe przetwarzanie**:
  - Upload i automatyczne rozpoznanie formatu pliku
  - Automatyczne generowanie słów kluczowych z możliwością edycji
  - Konfiguracja opcji przetwarzania i wybór formatu docelowego

- **System Prompt**:
  - Pole tekstowe dla kontekstu zadania (np. "Segregujesz stare artykuły weterynaryjne...")
  - Dostosowanie instrukcji do konkretnego zadania bez ingerencji w kod

- **Opcje modelu**:
  - Temperature (0.0-1.0)
  - Max tokens 
  - Top-p sampling
  - Frequency penalty

- **Zaawansowane opcje**:
  - Reasoning - włączenie/wyłączenie śladu rozumowania
  - Anonymization - anonimizacja danych wrażliwych
  - Token limit per chunk - kontrola długości fragmentów
  - Chunk overlap - nakładanie się fragmentów tekstu

### 2. Ulepszony Batch Processing

- **Pełna automatyzacja**:
  - Automatyczne przetwarzanie wielu plików bez potrzeby ręcznej edycji
  - Brak etapu edycji słów kluczowych - wszystko dzieje się automatycznie
  - Strategia "YoLo" - uruchom i zapomnij (domyślna)
  - Strategia "Paranoid" - podgląd co X plików (opcjonalna)

- **Multi-model Processing**:
  - Równoległe wykorzystanie kilku modeli (checkbox)
  - Strategie dystrybucji plików między modele:
    - Round Robin
    - File Size Based
    - File Type Based
    - Cost Optimization

- **Kontrola przetwarzania**:
  - Elastyczna konfiguracja maksymalnej liczby równoległych zadań
  - Szacowanie kosztów i ograniczenia budżetowe
  - Podgląd postępu z szacowanym czasem ukończenia
  - Możliwość wstrzymania/wznowienia procesu

### 3. Format pośredni JSON

```json
[
  {
    "instruction": "Pytanie/instrukcja",
    "input": "Kontekst z dokumentu",
    "output": "Odpowiedź/wynik",
    "metadata": {
      "source_file": "plik_źródłowy.pdf",
      "chunk_index": 3,
      "total_chunks": 12,
      "model_used": "claude-3-opus-20240229",
      "processing_time": "1.23s",
      "confidence_score": 0.94,
      "keywords": ["hemodializa", "ultrafiltracja", "dyfuzja", "weterynaria"],
      "extracted_entities": ["hemodializa", "dializator", "osocze"]
    },
    "reasoning": "Analizując artykuł, widzę że opisuje hemodializę jako procedurę..."
  }
]
```

### 4. Nowa zakładka "Prepare Training Data"

- **Źródło danych**:
  - Wybór wcześniej przetworzonych batchów
  - Łączenie wielu batchów w jeden zestaw treningowy

- **Formaty wyjściowe**:
  - JSONL (standardowy)
  - Hugging Face Dataset
  - TFRecord (TensorFlow)
  - CSV/TSV
  - Parquet

- **Transformacje danych**:
  - Filtrowanie przykładów o niskiej jakości
  - Deduplikacja
  - Augmentacja danych
  - Balansowanie różnych kategorii

- **Podział danych**:
  - Train/Validation/Test split (konfigurowalne proporcje)
  - Stratyfikacja podziału
  - Cross-validation

- **Raportowanie i analiza**:
  - Statystyki zestawu
  - Wizualizacje rozkładów
  - Raport potencjalnych problemów

### 5. Rozszerzenie o multimodalność (podejście hybrydowe)

- **W pierwszej fazie - integracja z AnyDataset**:
  - Podstawowe przetwarzanie obrazów i audio
  - Rozszerzony format JSON obsługujący wielomodalność
  - Nowa zakładka "Multimedia" dla uploadów obrazów i audio

- **Funkcje dla obrazów (VLM)**:
  - Automatyczne tagowanie i opis
  - Detekcja obiektów i segmentacja
  - Ekstrakcja regionów zainteresowania

- **Funkcje dla audio/głosu**:
  - Automatyczna transkrypcja
  - Ekstrakcja parametrów dźwięku
  - Normalizacja i poprawa jakości

- **Architektura modułowa**:
  - Możliwość wydzielenia specjalistycznych narzędzi w przyszłości
  - Standardowy format danych dla komunikacji między modułami

## Kolejne kroki implementacyjne

1. **Faza 1** - Ulepszenie interfejsu Process File:
   - Wdrożenie nowego interfejsu trójstopniowego
   - Dodanie opcji reasoningu i dynamicznej konfiguracji modelu
   - Integracja z automatycznym generowaniem słów kluczowych

2. **Faza 2** - Modernizacja Batch Processing:
   - Implementacja "YoLo" i "Paranoid" strategii
   - Dodanie wielomodelowego przetwarzania
   - Rozbudowa funkcji kontroli kosztów i zasobów

3. **Faza 3** - Zakładka Prepare Training Data:
   - Implementacja przygotowania danych treningowych
   - Dodanie transformacji i walidacji danych
   - Raportowanie i wizualizacje

4. **Faza 4** - Podstawowe wsparcie dla multimodalności:
   - Rozszerzenie formatu JSON
   - Podstawowe przetwarzanie obrazów i audio
   - Zakładka "Multimedia"

## Wizja długoterminowa

W dłuższej perspektywie AnyDataset może ewoluować w kierunku kompleksowej platformy przygotowania danych dla AI, obsługującej wszystkie modalności, z zaawansowanymi funkcjami walidacji jakości, augmentacji i wizualizacji. System modułowy pozwoli na elastyczne dostosowanie do różnych przypadków użycia, zachowując jednocześnie spójne doświadczenie użytkownika.

Priorytetem jest utrzymanie prostoty i intuicyjności interfejsu, przy jednoczesnym zapewnieniu zaawansowanych funkcji dla doświadczonych użytkowników.