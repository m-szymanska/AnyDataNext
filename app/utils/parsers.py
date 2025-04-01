import os
import json

def parse_txt(file_path: str, logger) -> list:

    logger.info(f"[parse_txt] Parsing .txt file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except Exception as e:
        logger.error(f"[parse_txt] Błąd odczytu pliku .txt: {file_path}. Szczegóły: {e}")
        raise

    # Minimalistycznie - tworzymy 1 rekord z całym tekstem w polu 'input'.
    parsed_data = [{
        "instruction": "",
        "input": text_content,
        "output": ""
    }]
    return parsed_data


def parse_json_file(file_path: str, logger) -> list:

    logger.info(f"[parse_json_file] Parsing .json file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Zakładamy, że data to list, ale można dodać weryfikację.
            if not isinstance(data, list):
                logger.warning("[parse_json_file] Oczekiwano listy, a otrzymano inną strukturę.")
                # Możesz tu dodać dodatkową logikę przerabiania obiektu
            return data
    except Exception as e:
        logger.error(f"[parse_json_file] Błąd parsowania pliku .json: {file_path}. Szczegóły: {e}")
        raise


def parse_jsonl_file(file_path: str, logger) -> list:

    logger.info(f"[parse_jsonl_file] Parsing .jsonl file: {file_path}")
    parsed_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue  # pomijamy puste linie
                try:
                    obj = json.loads(line)
                    parsed_data.append(obj)
                except json.JSONDecodeError as decode_err:
                    logger.error(f"[parse_jsonl_file] Błąd parsowania w linii {line_num}: {decode_err}")
                    # Możesz tu podjąć decyzję, czy rzucić wyjątek, czy pominąć linię.
                    raise
        return parsed_data
    except Exception as e:
        logger.error(f"[parse_jsonl_file] Błąd odczytu pliku .jsonl: {file_path}. Szczegóły: {e}")
        raise


def parse_file(file_path: str, logger) -> list:

    logger.info(f"[parse_file] Rozpoczynam parsowanie pliku: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return parse_txt(file_path, logger)
    elif ext == ".json":
        return parse_json_file(file_path, logger)
    elif ext == ".jsonl":
        return parse_jsonl_file(file_path, logger)
    else:
        msg = f"Nieobsługiwane rozszerzenie: {ext}. Możliwe: .txt, .json, .jsonl"
        logger.error(f"[parse_file] {msg}")
        raise ValueError(msg)
