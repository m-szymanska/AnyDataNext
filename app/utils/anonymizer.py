"""
Utilities for detecting and anonymizing personally identifiable information (PII).
"""

import re


def detect_pii(text):
    """
    Detects potential PII (Personally Identifiable Information) in text.
    
    Args:
        text (str): The text to check for PII
        
    Returns:
        dict: Dictionary with PII type as keys and a list of tuples 
              (start, end, value) for each match
    """
    patterns = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(?:\+\d{1,3}[- ]?)?\(?\d{2,3}\)?[- ]?\d{3,4}[- ]?\d{3,4}\b',
        "pesel": r'\b\d{11}\b',
        "address": r'\b(ul\.|ulica|al\.|aleja)[^,\n]{5,40}[\d]+[a-zA-Z]?\/[\d]+[a-zA-Z]?\b',
        "nip": r'\b\d{3}[-]?\d{3}[-]?\d{2}[-]?\d{2}\b',
        "regon": r'\b\d{9}|(\d{14})\b',
        "name_surname": r'\b[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+ [A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+(?:\-[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]+)?\b'
    }
    
    findings = {}
    for pii_type, pattern in patterns.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            if pii_type not in findings:
                findings[pii_type] = []
            findings[pii_type].append((match.start(), match.end(), match.group()))
    
    return findings


def anonymize_text(text, pii_findings=None):
    """
    Anonymizes PII in text by replacing with placeholders.
    
    Args:
        text (str): The text to anonymize
        pii_findings (dict, optional): Dictionary of PII findings from detect_pii
            If None, detect_pii will be called
    
    Returns:
        str: Anonymized text
    """
    if pii_findings is None:
        pii_findings = detect_pii(text)
    
    # Sort findings from end to start to avoid changing indexes when replacing
    replacements = []
    for pii_type, findings in pii_findings.items():
        for start, end, value in findings:
            replacements.append((start, end, value, pii_type))
    
    replacements.sort(reverse=True, key=lambda x: x[0])
    
    # Replace PII with placeholders
    anonymized_text = text
    for start, end, value, pii_type in replacements:
        replacement = f"[{pii_type.upper()}]"
        anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]
    
    return anonymized_text