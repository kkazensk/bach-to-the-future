#!/usr/bin/env python3

'''
Purpose of this document is to convert the MusicXML file to the PDF format using the Music21 Library. This needs to be done so the GUI can then view the PDF
'''

#credit to this stack overflow link that told me it was possible to do the conversion with the library.https://stackoverflow.com/questions/22883594/generating-pdf-midi-from-musicxml

from music21 import converter
import os

def convert_musicxml_to_pdf(xml_path, pdf_path, fdname):
    score = converter.parse(xml_path)
    score.write(fdname+'.pdf', fp=pdf_path)  # You can also use 'musicxml.pdf' if preferred