"""Standalone PDF power tools: Compress, Edit (with OCR), and Preflight.

These are independent of the cropper pipeline. Each tool is a thin service that
operates on raw PDF bytes with PyMuPDF (``fitz``) so the work stays fast and has
no external binary dependency beyond the optional Tesseract used for OCR.
"""
