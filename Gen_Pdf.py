from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# Nom du fichier PDF
pdf_file = "stress_report.pdf"

# Contenu : titre + paragraphe
title = "Rapport de Stress"
paragraph_text = """
John has been feeling overwhelmed at work. He struggles to sleep at night and constantly worries about deadlines. 
Every day, he feels tense and fatigued, showing classic signs of stress. His friends have noticed his mood swings and irritability. 
It's important for him to find coping strategies and seek support to manage stress effectively.
"""

# Création du document
doc = SimpleDocTemplate(pdf_file, pagesize=A4,
                        rightMargin=2*cm, leftMargin=2*cm,
                        topMargin=2*cm, bottomMargin=2*cm)

# Styles pour le texte
styles = getSampleStyleSheet()
story = []

# Ajouter le titre
story.append(Paragraph(title, styles['Title']))
story.append(Spacer(1, 12))

# Ajouter le paragraphe (justifié)
story.append(Paragraph(paragraph_text, styles['BodyText']))

# Générer le PDF
doc.build(story)

print(f"✅ PDF généré avec succès : {pdf_file}")
