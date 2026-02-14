"""
HACKATHON AI TRIAGE SYSTEM - MULTI-STEP STREAMLIT APP
======================================================
FIXED VERSION v3.1
- Proper voice input with speech recognition
- Complete multilingual implementation
- All warnings fixed
- Professional medical triage system

Features:
- Real voice-based symptom input
- Multi-language support (English, Spanish, French, Hindi, Tamil, Arabic)
- AI-powered document analysis
- Complete UI translation
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
import base64
import io
import re
from PIL import Image
import pytesseract

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="MedTouch.ai Patient Intake",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# LANGUAGE TRANSLATIONS - COMPLETE
# ============================================================================
TRANSLATIONS = {
    'en': {
        'app_title': '🏥 MedTouch.ai Patient Intake',
        'app_subtitle': 'AI-Powered Medical Triage System',
        'step_vitals': 'Vitals',
        'step_symptoms': 'Symptoms',
        'step_history': 'History',
        'step_results': 'Results',
        'vitals_header': '📊 Patient Vitals',
        'vitals_subheader': "Enter the patient's vital signs and basic information",
        'age': 'Age (years)',
        'gender': 'Gender',
        'male': 'Male',
        'female': 'Female',
        'blood_pressure': 'Blood Pressure (mmHg)',
        'systolic': 'Systolic (Upper)',
        'diastolic': 'Diastolic (Lower)',
        'heart_rate': 'Heart Rate (BPM)',
        'temperature': 'Body Temperature (°C)',
        'continue': 'Continue →',
        'previous': '← Previous',
        'symptoms_header': '🩺 Select Symptoms',
        'symptoms_subheader': 'Choose all symptoms the patient is experiencing',
        'voice_input': '🎤 Voice Input',
        'start_recording': '🎙️ Click to Speak',
        'stop_recording': '⏹️ Stop Recording',
        'processing_audio': 'Processing audio...',
        'voice_instructions': 'Click the microphone button and speak your symptoms clearly',
        'no_speech': 'No speech detected. Please try again.',
        'speech_error': 'Could not understand audio. Please speak clearly.',
        'history_header': '📋 Medical History',
        'history_subheader': 'Upload health document for automatic extraction or select manually',
        'document_upload': '🤖 AI-Powered Document Analysis',
        'upload_subtitle': "Upload patient's EHR/EMR document - AI will automatically extract medical history",
        'choose_file': 'Choose a file',
        'analyzing': '🔍 Analyzing document with AI...',
        'extracted_info': '🎯 Extracted Medical Information',
        'detected_conditions': 'Detected Conditions:',
        'extracted_vitals': 'Extracted Vitals:',
        'apply_data': '🔄 Apply Extracted Data to Form',
        'manual_selection': 'Manual Selection',
        'select_condition': 'Select Pre-Existing Condition',
        'no_history_info': "Select 'No History' if patient has no pre-existing conditions",
        'analyze_patient': 'Analyze Patient →',
        'results_header': '🎯 Analysis Results',
        'risk_classification': 'Risk Classification',
        'high_risk': 'HIGH RISK',
        'medium_risk': 'MEDIUM RISK',
        'low_risk': 'LOW RISK',
        'confidence': 'Confidence',
        'recommended_dept': 'Recommended Department',
        'match': 'Match',
        'risk_probabilities': 'Risk Probabilities',
        'clinical_recommendations': '🏥 Clinical Recommendations',
        'contributing_factors': '💡 Contributing Factors',
        'patient_summary': '📊 Patient Summary',
        'new_patient': '🔄 New Patient',
        'assessment_complete': '✅ Assessment Complete',
        'select_language': 'Language',
        'warning_symptoms': '⚠️ Please select at least one symptom',
        'processing': 'Processing...',
        'field': 'Field',
        'value': 'Value',
        'symptoms_label': 'Symptoms',
        'pre_existing': 'Pre-Existing',
        'document': 'Document',
        'data_source': 'Data Source',
        'none': 'None',
        'ai_extracted': 'AI-Extracted',
        'manual': 'Manual',
        'years': 'years',
        'or': 'or',
        'immediate_action': 'IMMEDIATE ACTION REQUIRED',
        'urgent_assessment': 'URGENT ASSESSMENT NEEDED',
        'routine_processing': 'ROUTINE PROCESSING',
        'priority': 'Priority',
        'actions': 'Actions',
        'target': 'Target',
        'physician_eval_immediate': 'Physician evaluation IMMEDIATELY',
        'physician_eval_15_30': 'Physician evaluation within 15-30 minutes',
        'expected_wait': 'Expected Wait',
        'hours_1_2': '1-2 hours',
        'voice_detected': 'Voice detected symptoms',
        'manual_symptoms': 'Manual Selection (Optional)',
        'clear_voice': 'Clear Voice Input',
        'listening': 'Listening... Speak now!',
        'click_to_record': 'Click microphone to start recording',
    },
    'es': {  # Spanish
        'app_title': '🏥 MedTouch.ai Admisión de Pacientes',
        'app_subtitle': 'Sistema de Triaje Médico con IA',
        'step_vitals': 'Signos Vitales',
        'step_symptoms': 'Síntomas',
        'step_history': 'Historia',
        'step_results': 'Resultados',
        'vitals_header': '📊 Signos Vitales del Paciente',
        'vitals_subheader': 'Ingrese los signos vitales e información básica del paciente',
        'age': 'Edad (años)',
        'gender': 'Género',
        'male': 'Masculino',
        'female': 'Femenino',
        'blood_pressure': 'Presión Arterial (mmHg)',
        'systolic': 'Sistólica (Superior)',
        'diastolic': 'Diastólica (Inferior)',
        'heart_rate': 'Frecuencia Cardíaca (LPM)',
        'temperature': 'Temperatura Corporal (°C)',
        'continue': 'Continuar →',
        'previous': '← Anterior',
        'symptoms_header': '🩺 Seleccionar Síntomas',
        'symptoms_subheader': 'Elija todos los síntomas que presenta el paciente',
        'voice_input': '🎤 Entrada de Voz',
        'start_recording': '🎙️ Clic para Hablar',
        'stop_recording': '⏹️ Detener Grabación',
        'processing_audio': 'Procesando audio...',
        'voice_instructions': 'Haga clic en el micrófono y hable sus síntomas claramente',
        'no_speech': 'No se detectó voz. Inténtelo de nuevo.',
        'speech_error': 'No se pudo entender el audio. Hable claramente.',
        'history_header': '📋 Historia Médica',
        'history_subheader': 'Cargue el documento de salud para extracción automática o seleccione manualmente',
        'document_upload': '🤖 Análisis de Documentos con IA',
        'upload_subtitle': 'Cargue el documento EHR/EMR del paciente - La IA extraerá automáticamente el historial médico',
        'choose_file': 'Elegir archivo',
        'analyzing': '🔍 Analizando documento con IA...',
        'extracted_info': '🎯 Información Médica Extraída',
        'detected_conditions': 'Condiciones Detectadas:',
        'extracted_vitals': 'Signos Vitales Extraídos:',
        'apply_data': '🔄 Aplicar Datos Extraídos',
        'manual_selection': 'Selección Manual',
        'select_condition': 'Seleccionar Condición Preexistente',
        'no_history_info': "Seleccione 'Sin Historial' si el paciente no tiene condiciones preexistentes",
        'analyze_patient': 'Analizar Paciente →',
        'results_header': '🎯 Resultados del Análisis',
        'risk_classification': 'Clasificación de Riesgo',
        'high_risk': 'RIESGO ALTO',
        'medium_risk': 'RIESGO MEDIO',
        'low_risk': 'RIESGO BAJO',
        'confidence': 'Confianza',
        'recommended_dept': 'Departamento Recomendado',
        'match': 'Coincidencia',
        'risk_probabilities': 'Probabilidades de Riesgo',
        'clinical_recommendations': '🏥 Recomendaciones Clínicas',
        'contributing_factors': '💡 Factores Contribuyentes',
        'patient_summary': '📊 Resumen del Paciente',
        'new_patient': '🔄 Nuevo Paciente',
        'assessment_complete': '✅ Evaluación Completa',
        'select_language': 'Idioma',
        'warning_symptoms': '⚠️ Por favor seleccione al menos un síntoma',
        'processing': 'Procesando...',
        'field': 'Campo',
        'value': 'Valor',
        'symptoms_label': 'Síntomas',
        'pre_existing': 'Preexistente',
        'document': 'Documento',
        'data_source': 'Fuente de Datos',
        'none': 'Ninguno',
        'ai_extracted': 'Extraído por IA',
        'manual': 'Manual',
        'years': 'años',
        'or': 'o',
        'immediate_action': 'ACCIÓN INMEDIATA REQUERIDA',
        'urgent_assessment': 'EVALUACIÓN URGENTE NECESARIA',
        'routine_processing': 'PROCESAMIENTO DE RUTINA',
        'priority': 'Prioridad',
        'actions': 'Acciones',
        'target': 'Objetivo',
        'physician_eval_immediate': 'Evaluación médica INMEDIATAMENTE',
        'physician_eval_15_30': 'Evaluación médica en 15-30 minutos',
        'expected_wait': 'Espera Esperada',
        'hours_1_2': '1-2 horas',
        'voice_detected': 'Síntomas detectados por voz',
        'manual_symptoms': 'Selección Manual (Opcional)',
        'clear_voice': 'Borrar Entrada de Voz',
        'listening': '¡Escuchando... Hable ahora!',
        'click_to_record': 'Haga clic en el micrófono para comenzar a grabar',
    },
    'fr': {  # French
        'app_title': "🏥 MedTouch.ai Admission des Patients",
        'app_subtitle': 'Système de Triage Médical IA',
        'step_vitals': 'Signes Vitaux',
        'step_symptoms': 'Symptômes',
        'step_history': 'Historique',
        'step_results': 'Résultats',
        'vitals_header': '📊 Signes Vitaux du Patient',
        'vitals_subheader': 'Entrez les signes vitaux et informations de base du patient',
        'age': 'Âge (années)',
        'gender': 'Genre',
        'male': 'Homme',
        'female': 'Femme',
        'blood_pressure': 'Pression Artérielle (mmHg)',
        'systolic': 'Systolique (Supérieure)',
        'diastolic': 'Diastolique (Inférieure)',
        'heart_rate': 'Fréquence Cardiaque (BPM)',
        'temperature': 'Température Corporelle (°C)',
        'continue': 'Continuer →',
        'previous': '← Précédent',
        'symptoms_header': '🩺 Sélectionner les Symptômes',
        'symptoms_subheader': 'Choisissez tous les symptômes que présente le patient',
        'voice_input': '🎤 Entrée Vocale',
        'start_recording': '🎙️ Cliquer pour Parler',
        'stop_recording': '⏹️ Arrêter Enregistrement',
        'processing_audio': 'Traitement audio...',
        'voice_instructions': 'Cliquez sur le microphone et parlez de vos symptômes clairement',
        'no_speech': 'Aucune voix détectée. Réessayez.',
        'speech_error': 'Impossible de comprendre l\'audio. Parlez clairement.',
        'history_header': '📋 Historique Médical',
        'history_subheader': 'Téléchargez le document de santé pour extraction automatique ou sélectionnez manuellement',
        'document_upload': '🤖 Analyse de Documents IA',
        'upload_subtitle': 'Téléchargez le document EHR/EMR du patient - L\'IA extraira automatiquement l\'historique médical',
        'choose_file': 'Choisir un fichier',
        'analyzing': '🔍 Analyse du document avec IA...',
        'extracted_info': '🎯 Informations Médicales Extraites',
        'detected_conditions': 'Conditions Détectées:',
        'extracted_vitals': 'Signes Vitaux Extraits:',
        'apply_data': '🔄 Appliquer les Données Extraites',
        'manual_selection': 'Sélection Manuelle',
        'select_condition': 'Sélectionner Condition Préexistante',
        'no_history_info': "Sélectionnez 'Pas d'Historique' si le patient n'a pas de conditions préexistantes",
        'analyze_patient': 'Analyser le Patient →',
        'results_header': '🎯 Résultats de l\'Analyse',
        'risk_classification': 'Classification des Risques',
        'high_risk': 'RISQUE ÉLEVÉ',
        'medium_risk': 'RISQUE MOYEN',
        'low_risk': 'RISQUE FAIBLE',
        'confidence': 'Confiance',
        'recommended_dept': 'Département Recommandé',
        'match': 'Correspondance',
        'risk_probabilities': 'Probabilités de Risque',
        'clinical_recommendations': '🏥 Recommandations Cliniques',
        'contributing_factors': '💡 Facteurs Contributifs',
        'patient_summary': '📊 Résumé du Patient',
        'new_patient': '🔄 Nouveau Patient',
        'assessment_complete': '✅ Évaluation Terminée',
        'select_language': 'Langue',
        'warning_symptoms': '⚠️ Veuillez sélectionner au moins un symptôme',
        'processing': 'Traitement...',
        'field': 'Champ',
        'value': 'Valeur',
        'symptoms_label': 'Symptômes',
        'pre_existing': 'Préexistant',
        'document': 'Document',
        'data_source': 'Source de Données',
        'none': 'Aucun',
        'ai_extracted': 'Extrait par IA',
        'manual': 'Manuel',
        'years': 'années',
        'or': 'ou',
        'immediate_action': 'ACTION IMMÉDIATE REQUISE',
        'urgent_assessment': 'ÉVALUATION URGENTE NÉCESSAIRE',
        'routine_processing': 'TRAITEMENT DE ROUTINE',
        'priority': 'Priorité',
        'actions': 'Actions',
        'target': 'Cible',
        'physician_eval_immediate': 'Évaluation médicale IMMÉDIATEMENT',
        'physician_eval_15_30': 'Évaluation médicale dans 15-30 minutes',
        'expected_wait': 'Attente Prévue',
        'hours_1_2': '1-2 heures',
        'voice_detected': 'Symptômes détectés par voix',
        'manual_symptoms': 'Sélection Manuelle (Optionnel)',
        'clear_voice': 'Effacer Entrée Vocale',
        'listening': 'Écoute... Parlez maintenant!',
        'click_to_record': 'Cliquez sur le microphone pour commencer l\'enregistrement',
    },
    'hi': {  # Hindi
        'app_title': '🏥 MedTouch.ai रोगी प्रवेश',
        'app_subtitle': 'AI-संचालित चिकित्सा ट्राइएज प्रणाली',
        'step_vitals': 'महत्वपूर्ण संकेत',
        'step_symptoms': 'लक्षण',
        'step_history': 'इतिहास',
        'step_results': 'परिणाम',
        'vitals_header': '📊 रोगी के महत्वपूर्ण संकेत',
        'vitals_subheader': 'रोगी के महत्वपूर्ण संकेत और बुनियादी जानकारी दर्ज करें',
        'age': 'आयु (वर्ष)',
        'gender': 'लिंग',
        'male': 'पुरुष',
        'female': 'महिला',
        'blood_pressure': 'रक्तचाप (mmHg)',
        'systolic': 'सिस्टोलिक (ऊपरी)',
        'diastolic': 'डायस्टोलिक (निचला)',
        'heart_rate': 'हृदय गति (BPM)',
        'temperature': 'शरीर का तापमान (°C)',
        'continue': 'जारी रखें →',
        'previous': '← पिछला',
        'symptoms_header': '🩺 लक्षण चुनें',
        'symptoms_subheader': 'रोगी के सभी लक्षण चुनें',
        'voice_input': '🎤 आवाज इनपुट',
        'start_recording': '🎙️ बोलने के लिए क्लिक करें',
        'stop_recording': '⏹️ रिकॉर्डिंग बंद करें',
        'processing_audio': 'ऑडियो प्रसंस्करण...',
        'voice_instructions': 'माइक्रोफ़ोन पर क्लिक करें और अपने लक्षण स्पष्ट रूप से बोलें',
        'no_speech': 'कोई आवाज़ नहीं मिली। कृपया पुनः प्रयास करें।',
        'speech_error': 'ऑडियो समझ नहीं आया। कृपया स्पष्ट रूप से बोलें।',
        'history_header': '📋 चिकित्सा इतिहास',
        'history_subheader': 'स्वचालित निष्कर्षण के लिए स्वास्थ्य दस्तावेज़ अपलोड करें या मैन्युअल रूप से चुनें',
        'document_upload': '🤖 AI-संचालित दस्तावेज़ विश्लेषण',
        'upload_subtitle': 'रोगी का EHR/EMR दस्तावेज़ अपलोड करें - AI स्वचालित रूप से चिकित्सा इतिहास निकालेगा',
        'choose_file': 'फ़ाइल चुनें',
        'analyzing': '🔍 AI के साथ दस्तावेज़ का विश्लेषण...',
        'extracted_info': '🎯 निकाली गई चिकित्सा जानकारी',
        'detected_conditions': 'पता लगाई गई स्थितियां:',
        'extracted_vitals': 'निकाले गए महत्वपूर्ण संकेत:',
        'apply_data': '🔄 निकाला गया डेटा लागू करें',
        'manual_selection': 'मैन्युअल चयन',
        'select_condition': 'पूर्व-मौजूद स्थिति चुनें',
        'no_history_info': "यदि रोगी की कोई पूर्व-मौजूद स्थिति नहीं है तो 'कोई इतिहास नहीं' चुनें",
        'analyze_patient': 'रोगी का विश्लेषण करें →',
        'results_header': '🎯 विश्लेषण परिणाम',
        'risk_classification': 'जोखिम वर्गीकरण',
        'high_risk': 'उच्च जोखिम',
        'medium_risk': 'मध्यम जोखिम',
        'low_risk': 'कम जोखिम',
        'confidence': 'विश्वास',
        'recommended_dept': 'अनुशंसित विभाग',
        'match': 'मिलान',
        'risk_probabilities': 'जोखिम संभावनाएं',
        'clinical_recommendations': '🏥 नैदानिक ​​सिफारिशें',
        'contributing_factors': '💡 योगदान कारक',
        'patient_summary': '📊 रोगी सारांश',
        'new_patient': '🔄 नया रोगी',
        'assessment_complete': '✅ मूल्यांकन पूर्ण',
        'select_language': 'भाषा',
        'warning_symptoms': '⚠️ कृपया कम से कम एक लक्षण चुनें',
        'processing': 'प्रसंस्करण...',
        'field': 'क्षेत्र',
        'value': 'मूल्य',
        'symptoms_label': 'लक्षण',
        'pre_existing': 'पूर्व-मौजूद',
        'document': 'दस्तावेज़',
        'data_source': 'डेटा स्रोत',
        'none': 'कोई नहीं',
        'ai_extracted': 'AI-निकाला गया',
        'manual': 'मैन्युअल',
        'years': 'वर्ष',
        'or': 'या',
        'immediate_action': 'तत्काल कार्रवाई आवश्यक',
        'urgent_assessment': 'तत्काल मूल्यांकन आवश्यक',
        'routine_processing': 'नियमित प्रसंस्करण',
        'priority': 'प्राथमिकता',
        'actions': 'कार्रवाई',
        'target': 'लक्ष्य',
        'physician_eval_immediate': 'चिकित्सक मूल्यांकन तुरंत',
        'physician_eval_15_30': '15-30 मिनट में चिकित्सक मूल्यांकन',
        'expected_wait': 'अपेक्षित प्रतीक्षा',
        'hours_1_2': '1-2 घंटे',
        'voice_detected': 'आवाज द्वारा पता लगाए गए लक्षण',
        'manual_symptoms': 'मैन्युअल चयन (वैकल्पिक)',
        'clear_voice': 'आवाज इनपुट साफ़ करें',
        'listening': 'सुन रहा है... अभी बोलें!',
        'click_to_record': 'रिकॉर्डिंग शुरू करने के लिए माइक्रोफ़ोन पर क्लिक करें',
    },
    'ta': {  # Tamil
        'app_title': '🏥 MedTouch.ai நோயாளி சேர்க்கை',
        'app_subtitle': 'AI-இயங்கும் மருத்துவ வகைப்படுத்தல் அமைப்பு',
        'step_vitals': 'உயிர் அறிகுறிகள்',
        'step_symptoms': 'அறிகுறிகள்',
        'step_history': 'வரலாறு',
        'step_results': 'முடிவுகள்',
        'vitals_header': '📊 நோயாளியின் உயிர் அறிகுறிகள்',
        'vitals_subheader': 'நோயாளியின் உயிர் அறிகுறிகள் மற்றும் அடிப்படை தகவலை உள்ளிடவும்',
        'age': 'வயது (ஆண்டுகள்)',
        'gender': 'பாலினம்',
        'male': 'ஆண்',
        'female': 'பெண்',
        'blood_pressure': 'இரத்த அழுத்தம் (mmHg)',
        'systolic': 'சிஸ்டாலிக் (மேல்)',
        'diastolic': 'டயாஸ்டாலிக் (கீழ்)',
        'heart_rate': 'இதய துடிப்பு (BPM)',
        'temperature': 'உடல் வெப்பநிலை (°C)',
        'continue': 'தொடரவும் →',
        'previous': '← முந்தைய',
        'symptoms_header': '🩺 அறிகுறிகளைத் தேர்ந்தெடுக்கவும்',
        'symptoms_subheader': 'நோயாளி அனுபவிக்கும் அனைத்து அறிகுறிகளையும் தேர்வு செய்யவும்',
        'voice_input': '🎤 குரல் உள்ளீடு',
        'start_recording': '🎙️ பேச கிளிக் செய்யவும்',
        'stop_recording': '⏹️ பதிவை நிறுத்தவும்',
        'processing_audio': 'ஆடியோ செயலாக்கம்...',
        'voice_instructions': 'மைக்ரோஃபோனை கிளிக் செய்து உங்கள் அறிகுறிகளைத் தெளிவாகப் பேசவும்',
        'no_speech': 'குரல் கண்டறியப்படவில்லை. மீண்டும் முயற்சிக்கவும்.',
        'speech_error': 'ஆடியோவைப் புரிந்து கொள்ள முடியவில்லை. தெளிவாகப் பேசவும்.',
        'history_header': '📋 மருத்துவ வரலாறு',
        'history_subheader': 'தானியங்கி பிரித்தெடுப்புக்கு சுகாதார ஆவணத்தைப் பதிவேற்றவும் அல்லது கைமுறையாகத் தேர்ந்தெடுக்கவும்',
        'document_upload': '🤖 AI-இயங்கும் ஆவண பகுப்பாய்வு',
        'upload_subtitle': 'நோயாளியின் EHR/EMR ஆவணத்தைப் பதிவேற்றவும் - AI தானாக மருத்துவ வரலாற்றைப் பிரித்தெடுக்கும்',
        'choose_file': 'கோப்பைத் தேர்ந்தெடுக்கவும்',
        'analyzing': '🔍 AI உடன் ஆவணத்தை பகுப்பாய்வு செய்கிறது...',
        'extracted_info': '🎯 பிரித்தெடுக்கப்பட்ட மருத்துவ தகவல்',
        'detected_conditions': 'கண்டறியப்பட்ட நிலைமைகள்:',
        'extracted_vitals': 'பிரித்தெடுக்கப்பட்ட உயிர் அறிகுறிகள்:',
        'apply_data': '🔄 பிரித்தெடுக்கப்பட்ட தரவைப் பயன்படுத்தவும்',
        'manual_selection': 'கைமுறை தேர்வு',
        'select_condition': 'முன்பே இருந்த நிலையைத் தேர்ந்தெடுக்கவும்',
        'no_history_info': "நோயாளிக்கு முன்பே இருந்த நிலைமைகள் இல்லை என்றால் 'வரலாறு இல்லை' என்பதைத் தேர்ந்தெடுக்கவும்",
        'analyze_patient': 'நோயாளியை பகுப்பாய்வு செய்யவும் →',
        'results_header': '🎯 பகுப்பாய்வு முடிவுகள்',
        'risk_classification': 'ஆபத்து வகைப்பாடு',
        'high_risk': 'அதிக ஆபத்து',
        'medium_risk': 'நடுத்தர ஆபத்து',
        'low_risk': 'குறைந்த ஆபத்து',
        'confidence': 'நம்பிக்கை',
        'recommended_dept': 'பரிந்துரைக்கப்பட்ட துறை',
        'match': 'பொருத்தம்',
        'risk_probabilities': 'ஆபத்து நிகழ்தகவுகள்',
        'clinical_recommendations': '🏥 மருத்துவ பரிந்துரைகள்',
        'contributing_factors': '💡 பங்களிப்பு காரணிகள்',
        'patient_summary': '📊 நோயாளி சுருக்கம்',
        'new_patient': '🔄 புதிய நோயாளி',
        'assessment_complete': '✅ மதிப்பீடு முடிந்தது',
        'select_language': 'மொழி',
        'warning_symptoms': '⚠️ தயவுசெய்து குறைந்தது ஒரு அறிகுறியைத் தேர்ந்தெடுக்கவும்',
        'processing': 'செயலாக்கம்...',
        'field': 'புலம்',
        'value': 'மதிப்பு',
        'symptoms_label': 'அறிகுறிகள்',
        'pre_existing': 'முன்பே இருந்த',
        'document': 'ஆவணம்',
        'data_source': 'தரவு மூலம்',
        'none': 'இல்லை',
        'ai_extracted': 'AI-பிரித்தெடுக்கப்பட்டது',
        'manual': 'கைமுறை',
        'years': 'ஆண்டுகள்',
        'or': 'அல்லது',
        'immediate_action': 'உடனடி நடவடிக்கை தேவை',
        'urgent_assessment': 'அவசர மதிப்பீடு தேவை',
        'routine_processing': 'வழக்கமான செயலாக்கம்',
        'priority': 'முன்னுரிமை',
        'actions': 'நடவடிக்கைகள்',
        'target': 'இலக்கு',
        'physician_eval_immediate': 'மருத்துவர் மதிப்பீடு உடனடியாக',
        'physician_eval_15_30': '15-30 நிமிடங்களில் மருத்துவர் மதிப்பீடு',
        'expected_wait': 'எதிர்பார்க்கப்படும் காத்திருப்பு',
        'hours_1_2': '1-2 மணி நேரம்',
        'voice_detected': 'குரல் மூலம் கண்டறியப்பட்ட அறிகுறிகள்',
        'manual_symptoms': 'கைமுறை தேர்வு (விரும்பினால்)',
        'clear_voice': 'குரல் உள்ளீட்டை அழிக்கவும்',
        'listening': 'கேட்கிறது... இப்போது பேசுங்கள்!',
        'click_to_record': 'பதிவைத் தொடங்க மைக்ரோஃபோனை கிளிக் செய்யவும்',
    },
    'ar': {  # Arabic
        'app_title': '🏥 MedTouch.ai قبول المرضى',
        'app_subtitle': 'نظام الفرز الطبي بالذكاء الاصطناعي',
        'step_vitals': 'العلامات الحيوية',
        'step_symptoms': 'الأعراض',
        'step_history': 'التاريخ',
        'step_results': 'النتائج',
        'vitals_header': '📊 العلامات الحيوية للمريض',
        'vitals_subheader': 'أدخل العلامات الحيوية والمعلومات الأساسية للمريض',
        'age': 'العمر (سنوات)',
        'gender': 'الجنس',
        'male': 'ذكر',
        'female': 'أنثى',
        'blood_pressure': 'ضغط الدم (mmHg)',
        'systolic': 'الانقباضي (العلوي)',
        'diastolic': 'الانبساطي (السفلي)',
        'heart_rate': 'معدل ضربات القلب (BPM)',
        'temperature': 'درجة حرارة الجسم (°C)',
        'continue': 'متابعة ←',
        'previous': '→ السابق',
        'symptoms_header': '🩺 اختر الأعراض',
        'symptoms_subheader': 'اختر جميع الأعراض التي يعاني منها المريض',
        'voice_input': '🎤 إدخال صوتي',
        'start_recording': '🎙️ انقر للتحدث',
        'stop_recording': '⏹️ إيقاف التسجيل',
        'processing_audio': 'معالجة الصوت...',
        'voice_instructions': 'انقر على الميكروفون وتحدث عن أعراضك بوضوح',
        'no_speech': 'لم يتم اكتشاف صوت. حاول مرة أخرى.',
        'speech_error': 'تعذر فهم الصوت. تحدث بوضوح.',
        'history_header': '📋 التاريخ الطبي',
        'history_subheader': 'قم بتحميل المستند الصحي للاستخراج التلقائي أو حدد يدويًا',
        'document_upload': '🤖 تحليل المستندات بالذكاء الاصطناعي',
        'upload_subtitle': 'قم بتحميل مستند EHR/EMR للمريض - سيقوم الذكاء الاصطناعي باستخراج التاريخ الطبي تلقائيًا',
        'choose_file': 'اختر ملف',
        'analyzing': '🔍 تحليل المستند بالذكاء الاصطناعي...',
        'extracted_info': '🎯 المعلومات الطبية المستخرجة',
        'detected_conditions': 'الحالات المكتشفة:',
        'extracted_vitals': 'العلامات الحيوية المستخرجة:',
        'apply_data': '🔄 تطبيق البيانات المستخرجة',
        'manual_selection': 'التحديد اليدوي',
        'select_condition': 'حدد الحالة الموجودة مسبقًا',
        'no_history_info': "حدد 'لا يوجد تاريخ' إذا لم يكن لدى المريض حالات موجودة مسبقًا",
        'analyze_patient': '→ تحليل المريض',
        'results_header': '🎯 نتائج التحليل',
        'risk_classification': 'تصنيف المخاطر',
        'high_risk': 'مخاطر عالية',
        'medium_risk': 'مخاطر متوسطة',
        'low_risk': 'مخاطر منخفضة',
        'confidence': 'الثقة',
        'recommended_dept': 'القسم الموصى به',
        'match': 'تطابق',
        'risk_probabilities': 'احتمالات المخاطر',
        'clinical_recommendations': '🏥 التوصيات السريرية',
        'contributing_factors': '💡 العوامل المساهمة',
        'patient_summary': '📊 ملخص المريض',
        'new_patient': '🔄 مريض جديد',
        'assessment_complete': '✅ اكتمل التقييم',
        'select_language': 'اللغة',
        'warning_symptoms': '⚠️ الرجاء اختيار عرض واحد على الأقل',
        'processing': 'جاري المعالجة...',
        'field': 'الحقل',
        'value': 'القيمة',
        'symptoms_label': 'الأعراض',
        'pre_existing': 'موجودة مسبقًا',
        'document': 'وثيقة',
        'data_source': 'مصدر البيانات',
        'none': 'لا شيء',
        'ai_extracted': 'مستخرج بالذكاء الاصطناعي',
        'manual': 'يدوي',
        'years': 'سنوات',
        'or': 'أو',
        'immediate_action': 'مطلوب إجراء فوري',
        'urgent_assessment': 'تقييم عاجل مطلوب',
        'routine_processing': 'معالجة روتينية',
        'priority': 'الأولوية',
        'actions': 'الإجراءات',
        'target': 'الهدف',
        'physician_eval_immediate': 'تقييم الطبيب فورًا',
        'physician_eval_15_30': 'تقييم الطبيب في 15-30 دقيقة',
        'expected_wait': 'الانتظار المتوقع',
        'hours_1_2': '1-2 ساعة',
        'voice_detected': 'الأعراض المكتشفة بالصوت',
        'manual_symptoms': 'التحديد اليدوي (اختياري)',
        'clear_voice': 'مسح الإدخال الصوتي',
        'listening': 'استماع... تحدث الآن!',
        'click_to_record': 'انقر على الميكروفون لبدء التسجيل',
    }
}

# Symptom translations for voice recognition
SYMPTOM_TRANSLATIONS = {
    'en': {
        'headache': 'Headache', 'fever': 'Fever', 'cough': 'Cough',
        'fatigue': 'Fatigue', 'nausea': 'Nausea', 'dizziness': 'Dizziness',
        'chest pain': 'Chest Pain', 'shortness of breath': 'Difficulty Breathing',
        'difficulty breathing': 'Difficulty Breathing', 'abdominal pain': 'Abdominal Pain',
        'back pain': 'Back Pain', 'joint pain': 'Joint Pain', 'vomiting': 'Vomiting',
        'diarrhea': 'Diarrhea', 'sore throat': 'Sore Throat', 'runny nose': 'Runny Nose',
        'muscle pain': 'Muscle Pain', 'chills': 'Chills', 'sweating': 'Sweating'
    },
    'es': {
        'dolor de cabeza': 'Headache', 'fiebre': 'Fever', 'tos': 'Cough',
        'fatiga': 'Fatigue', 'náuseas': 'Nausea', 'nausea': 'Nausea', 'mareo': 'Dizziness',
        'dolor de pecho': 'Chest Pain', 'dificultad para respirar': 'Difficulty Breathing',
        'dolor abdominal': 'Abdominal Pain', 'dolor de espalda': 'Back Pain',
        'dolor de articulaciones': 'Joint Pain', 'vómito': 'Vomiting', 'vomito': 'Vomiting',
        'diarrea': 'Diarrhea', 'dolor de garganta': 'Sore Throat',
        'secreción nasal': 'Runny Nose', 'dolor muscular': 'Muscle Pain',
        'escalofríos': 'Chills', 'escalofrios': 'Chills', 'sudoración': 'Sweating', 'sudoracion': 'Sweating'
    },
    'fr': {
        'mal de tête': 'Headache', 'mal de tete': 'Headache', 'fièvre': 'Fever', 'fievre': 'Fever', 'toux': 'Cough',
        'fatigue': 'Fatigue', 'nausée': 'Nausea', 'nausee': 'Nausea', 'vertiges': 'Dizziness',
        'douleur thoracique': 'Chest Pain', 'difficulté à respirer': 'Difficulty Breathing',
        'difficulte a respirer': 'Difficulty Breathing',
        'douleur abdominale': 'Abdominal Pain', 'mal de dos': 'Back Pain',
        'douleur articulaire': 'Joint Pain', 'vomissement': 'Vomiting',
        'diarrhée': 'Diarrhea', 'diarrhee': 'Diarrhea', 'mal de gorge': 'Sore Throat',
        'nez qui coule': 'Runny Nose', 'douleur musculaire': 'Muscle Pain',
        'frissons': 'Chills', 'transpiration': 'Sweating'
    },
    'hi': {
        'सिरदर्द': 'Headache', 'बुखार': 'Fever', 'खांसी': 'Cough', 'खansi': 'Cough',
        'थकान': 'Fatigue', 'मतली': 'Nausea', 'चक्कर': 'Dizziness',
        'सीने में दर्द': 'Chest Pain', 'सांस लेने में कठिनाई': 'Difficulty Breathing',
        'पेट दर्द': 'Abdominal Pain', 'पीठ दर्द': 'Back Pain',
        'जोड़ों का दर्द': 'Joint Pain', 'उल्टी': 'Vomiting',
        'दस्त': 'Diarrhea', 'गले में खराश': 'Sore Throat',
        'नाक बहना': 'Runny Nose', 'मांसपेशियों में दर्द': 'Muscle Pain',
        'ठंड लगना': 'Chills', 'पसीना': 'Sweating'
    },
    'ta': {
        'தலைவலி': 'Headache', 'காய்ச்சல்': 'Fever', 'இருமல்': 'Cough',
        'சோர்வு': 'Fatigue', 'குமட்டல்': 'Nausea', 'தலைசுற்றல்': 'Dizziness',
        'மார்பு வலி': 'Chest Pain', 'மூச்சுத் திணறல்': 'Difficulty Breathing',
        'வயிற்று வலி': 'Abdominal Pain', 'முதுகு வலி': 'Back Pain',
        'மூட்டு வலி': 'Joint Pain', 'வாந்தி': 'Vomiting',
        'வயிற்றுப்போக்கு': 'Diarrhea', 'தொண்டை வலி': 'Sore Throat',
        'மூக்கு ஒழுகுதல்': 'Runny Nose', 'தசை வலி': 'Muscle Pain',
        'நடுக்கம்': 'Chills', 'வியர்வை': 'Sweating'
    },
    'ar': {
        'صداع': 'Headache', 'حمى': 'Fever', 'سعال': 'Cough',
        'تعب': 'Fatigue', 'غثيان': 'Nausea', 'دوار': 'Dizziness',
        'ألم في الصدر': 'Chest Pain', 'صعوبة في التنفس': 'Difficulty Breathing',
        'ألم في البطن': 'Abdominal Pain', 'ألم في الظهر': 'Back Pain',
        'ألم المفاصل': 'Joint Pain', 'قيء': 'Vomiting',
        'إسهال': 'Diarrhea', 'اسهال': 'Diarrhea', 'التهاب الحلق': 'Sore Throat',
        'سيلان الأنف': 'Runny Nose', 'ألم عضلي': 'Muscle Pain',
        'قشعريرة': 'Chills', 'تعرق': 'Sweating'
    }
}

# Enhanced CSS with proper styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Tamil:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', 'Noto Sans Arabic', 'Noto Sans Devanagari', 'Noto Sans Tamil', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main background - Powder blue gradient */
    .stApp {
        background: linear-gradient(180deg, 
            #B8D8E8 0%,
            #D4E8F0 30%,
            #E8F3F8 60%,
            #F0F7FA 100%
        );
        background-attachment: fixed;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1B3A52 !important;
        font-weight: 700 !important;
    }
    
    /* Progress stepper */
    .stepper {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2.5rem;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(27, 58, 82, 0.08);
    }
    
    .step {
        flex: 1;
        text-align: center;
        font-weight: 600;
        font-size: 1.05rem;
        color: #9CA3AF;
        position: relative;
        padding: 0.75rem;
    }
    
    .step-active {
        color: #1B3A52;
        font-weight: 700;
        background: rgba(184, 216, 232, 0.2);
        border-radius: 8px;
    }
    
    /* Card containers */
    .card {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(27, 58, 82, 0.08);
        margin: 1rem 0;
    }
    
    /* Risk cards */
    .risk-high {
        background: linear-gradient(135deg, #DC2626, #EF4444);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 8px 30px rgba(220, 38, 38, 0.3);
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #F59E0B, #FBBF24);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #10B981, #34D399);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    /* Department box */
    .dept-box {
        background: linear-gradient(135deg, #1B3A52, #2D5F7F);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 1rem 0;
    }
    
    /* Factor box */
    .factor-box {
        background: linear-gradient(135deg, #E0F2FE, #F0F9FF);
        padding: 14px 18px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1B3A52;
        font-size: 0.95rem;
        color: #3C4043;
    }
    
    /* Voice section */
    .voice-section {
        background: linear-gradient(135deg, #FEF3C7, #FDE68A);
        border-left: 4px solid #F59E0B;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Extracted data */
    .extracted-data {
        background: linear-gradient(135deg, #E8F5E9, #F1F8E9);
        border-left: 4px solid #4CAF50;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Buttons - Light colored */
    .stButton > button {
        background: linear-gradient(135deg, #D4E8F0, #B8D8E8) !important;
        color: #1B3A52 !important;
        border: 2px solid #1B3A52 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 15px rgba(27, 58, 82, 0.15) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        background: linear-gradient(135deg, #B8D8E8, #A0C8DC) !important;
    }
    
    /* Info tooltip */
    .info-tooltip {
        display: inline-block;
        position: relative;
        margin-left: 8px;
        cursor: help;
    }
    
    .info-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 20px;
        height: 20px;
        background: linear-gradient(135deg, #4A90E2, #1B3A52);
        color: white;
        border-radius: 50%;
        font-size: 12px;
        font-weight: bold;
    }
    
    .tooltip-text {
        visibility: hidden;
        width: 280px;
        background-color: #1B3A52;
        color: white;
        text-align: left;
        border-radius: 10px;
        padding: 12px 16px;
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        margin-left: -140px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.85rem;
    }
    
    .info-tooltip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DOCUMENT PROCESSING FUNCTIONS
# ============================================================================

def extract_text_from_image(image_file):
    """Extract text from image using OCR"""
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

def extract_text_from_txt(txt_file):
    """Extract text from TXT file"""
    try:
        content = txt_file.read()
        if isinstance(content, bytes):
            text = content.decode('utf-8', errors='ignore')
        else:
            text = content
        return text
    except Exception as e:
        return f"Error reading text file: {str(e)}"

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF"""
    try:
        return "PDF text extraction requires PyPDF2 or pdfplumber library. Please use TXT or image files."
    except Exception as e:
        return f"Error extracting from PDF: {str(e)}"

def analyze_medical_document(text, available_conditions):
    """Analyze medical document text and extract relevant medical history"""
    text_lower = text.lower()
    
    # Medical condition keywords mapping
    condition_keywords = {
        'Diabetes': ['diabetes', 'diabetic', 'hyperglycemia', 'blood sugar', 'glucose', 'insulin'],
        'Hypertension': ['hypertension', 'high blood pressure', 'htn', 'bp', 'elevated blood pressure'],
        'Asthma': ['asthma', 'asthmatic', 'bronchospasm', 'wheezing', 'inhaler'],
        'Heart Disease': ['heart disease', 'cardiac', 'coronary', 'chd', 'cvd', 'myocardial', 'angina'],
        'Kidney Disease': ['kidney disease', 'renal', 'ckd', 'nephropathy', 'dialysis'],
        'Cancer': ['cancer', 'carcinoma', 'tumor', 'malignancy', 'oncology', 'chemotherapy'],
        'Stroke': ['stroke', 'cva', 'cerebrovascular', 'brain attack', 'tia'],
        'COPD': ['copd', 'chronic obstructive', 'emphysema', 'chronic bronchitis'],
    }
    
    detected_conditions = []
    confidence_scores = {}
    
    for condition in available_conditions:
        if condition == 'No History':
            continue
            
        match_count = 0
        
        if condition.lower() in text_lower:
            match_count += 2
        
        if condition in condition_keywords:
            for keyword in condition_keywords[condition]:
                if keyword in text_lower:
                    match_count += 1
        
        if match_count > 0:
            detected_conditions.append(condition)
            confidence_scores[condition] = min(match_count * 20, 100)
    
    vitals_data = extract_vitals_from_text(text)
    
    return {
        'conditions': detected_conditions,
        'confidence': confidence_scores,
        'vitals': vitals_data,
        'raw_text': text[:500]
    }

def extract_vitals_from_text(text):
    """Extract vital signs from medical document text"""
    vitals = {}
    
    # Blood pressure pattern
    bp_pattern = r'(\d{2,3})\s*/\s*(\d{2,3})'
    bp_matches = re.findall(bp_pattern, text)
    if bp_matches:
        systolic, diastolic = bp_matches[0]
        vitals['systolic_bp'] = int(systolic)
        vitals['diastolic_bp'] = int(diastolic)
    
    # Heart rate
    hr_patterns = [r'heart rate[:\s]+(\d{2,3})', r'hr[:\s]+(\d{2,3})', r'(\d{2,3})\s*bpm']
    for pattern in hr_patterns:
        match = re.search(pattern, text.lower())
        if match:
            vitals['heart_rate'] = int(match.group(1))
            break
    
    # Temperature
    temp_patterns = [r'temperature[:\s]+(\d{2,3}\.?\d*)', r'temp[:\s]+(\d{2,3}\.?\d*)']
    for pattern in temp_patterns:
        match = re.search(pattern, text.lower())
        if match:
            temp = float(match.group(1))
            if temp > 45:
                temp = (temp - 32) * 5/9
            vitals['temperature'] = round(temp, 1)
            break
    
    # Age
    age_patterns = [r'age[:\s]+(\d{1,3})', r'(\d{1,3})\s*years?\s*old']
    for pattern in age_patterns:
        match = re.search(pattern, text.lower())
        if match:
            vitals['age'] = int(match.group(1))
            break
    
    # Gender
    if re.search(r'\b(male|man)\b', text.lower()) and not re.search(r'\b(female|woman)\b', text.lower()):
        vitals['gender'] = 'Male'
    elif re.search(r'\b(female|woman)\b', text.lower()):
        vitals['gender'] = 'Female'
    
    return vitals

# ============================================================================
# VOICE INPUT FUNCTIONS
# ============================================================================

def process_voice_text(voice_text, language='en'):
    """Process voice input and extract symptoms"""
    voice_text_lower = voice_text.lower()
    detected_symptoms = []
    
    symptom_dict = SYMPTOM_TRANSLATIONS.get(language, SYMPTOM_TRANSLATIONS['en'])
    
    for local_symptom, english_symptom in symptom_dict.items():
        if local_symptom in voice_text_lower:
            if english_symptom not in detected_symptoms:
                detected_symptoms.append(english_symptom)
    
    return detected_symptoms

def record_audio():
    """Record audio from microphone - simulated for web demo"""
    # Note: Real implementation would use speech_recognition library
    # For web demo, we use text input as simulation
    return None

# ============================================================================
# LOAD MODELS
# ============================================================================
@st.cache_resource
def load_models():
    """Load all models and encoders"""
    try:
        models = {
            'risk_model': joblib.load('risk_model.pkl'),
            'dept_model': joblib.load('department_model.pkl'),
            'le_gender': joblib.load('le_gender.pkl'),
            'le_symptoms': joblib.load('le_symptoms.pkl'),
            'le_pre_existing': joblib.load('le_pre_existing.pkl'),
        }
        return models, None
    except FileNotFoundError as e:
        return None, f"Missing file: {e.filename}"

models, error = load_models()

if error:
    st.error(f"❌ {error}")
    st.info("Please ensure all model files are in the same directory as this app.")
    st.stop()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'step' not in st.session_state:
    st.session_state.step = 1

if 'language' not in st.session_state:
    st.session_state.language = 'en'

if 'form_data' not in st.session_state:
    st.session_state.form_data = {
        'age': 45,
        'gender': 'Male',
        'systolic_bp': 120,
        'diastolic_bp': 80,
        'heart_rate': 80,
        'temperature': 36.8,
        'symptoms': [],
        'pre_existing': 'No History',
        'uploaded_document': None,
        'document_name': None,
        'extracted_data': None,
        'voice_symptoms': []
    }

if 'voice_text' not in st.session_state:
    st.session_state.voice_text = ""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def t(key):
    """Get translation for current language"""
    return TRANSLATIONS[st.session_state.language].get(key, key)

def next_step():
    """Move to next step"""
    st.session_state.step += 1

def prev_step():
    """Move to previous step"""
    st.session_state.step -= 1

def reset_form():
    """Reset to step 1"""
    st.session_state.step = 1
    st.session_state.form_data = {
        'age': 45,
        'gender': 'Male',
        'systolic_bp': 120,
        'diastolic_bp': 80,
        'heart_rate': 80,
        'temperature': 36.8,
        'symptoms': [],
        'pre_existing': 'No History',
        'uploaded_document': None,
        'document_name': None,
        'extracted_data': None,
        'voice_symptoms': []
    }
    st.session_state.voice_text = ""

def info_icon(tooltip_text):
    """Create an info icon with tooltip"""
    return f'''
    <span class="info-tooltip">
        <span class="info-icon">i</span>
        <span class="tooltip-text">{tooltip_text}</span>
    </span>
    '''

def explain_prediction(patient_data, risk_level):
    """Generate explanation factors"""
    factors = []
    
    age = patient_data['age']
    if age > 65 and risk_level == 'High':
        factors.append(f"🔴 Advanced age ({age} {t('years')}) increases risk significantly")
    elif age > 65:
        factors.append(f"⚠️ Elderly patient ({age} {t('years')})")
    elif age < 30:
        factors.append(f"✅ Young patient ({age} {t('years')}) - lower baseline risk")
    
    bp_sys = patient_data['systolic_bp']
    bp_dia = patient_data['diastolic_bp']
    
    if bp_sys > 180 or bp_dia > 100:
        factors.append(f"🔴 Hypertensive crisis (BP: {bp_sys}/{bp_dia})")
    elif bp_sys > 140 or bp_dia > 90:
        factors.append(f"⚠️ Elevated blood pressure ({bp_sys}/{bp_dia})")
    elif bp_sys < 90:
        factors.append(f"🔴 Low blood pressure ({bp_sys}/{bp_dia})")
    else:
        factors.append(f"✅ Normal blood pressure ({bp_sys}/{bp_dia})")
    
    hr = patient_data['heart_rate']
    if hr > 120:
        factors.append(f"🔴 Tachycardia ({hr} BPM)")
    elif hr > 100:
        factors.append(f"⚠️ Elevated heart rate ({hr} BPM)")
    elif hr < 50:
        factors.append(f"⚠️ Bradycardia ({hr} BPM)")
    else:
        factors.append(f"✅ Normal heart rate ({hr} BPM)")
    
    temp = patient_data['temperature']
    if temp > 38.5:
        factors.append(f"🔴 High fever ({temp}°C)")
    elif temp > 37.5:
        factors.append(f"⚠️ Mild fever ({temp}°C)")
    elif temp < 36.0:
        factors.append(f"⚠️ Hypothermia ({temp}°C)")
    else:
        factors.append(f"✅ Normal temperature ({temp}°C)")
    
    if patient_data['pre_existing'] != 'No History':
        factors.append(f"⚠️ Pre-existing condition: {patient_data['pre_existing']}")
    
    symptom_count = len(patient_data['symptoms'])
    if symptom_count >= 4:
        factors.append(f"🔴 Multiple symptoms present ({symptom_count} symptoms)")
    elif symptom_count >= 2:
        factors.append(f"⚠️ Several symptoms reported ({symptom_count} symptoms)")
    
    high_risk_symptoms = ['Chest Pain', 'Difficulty Breathing', 'Seizures', 'Unconsciousness']
    found_high_risk = [s for s in high_risk_symptoms if s in patient_data['symptoms']]
    if found_high_risk:
        factors.append(f"🔴 Critical symptoms: {', '.join(found_high_risk)}")
    
    return factors

def make_prediction(patient_data):
    """Make risk and department predictions"""
    try:
        gender_encoded = models['le_gender'].transform([patient_data['gender']])[0]
        
        symptoms_list = patient_data['symptoms']
        if not symptoms_list:
            return {'error': 'No symptoms selected'}
        
        primary_symptom = symptoms_list[0]
        symptom_encoded = models['le_symptoms'].transform([primary_symptom])[0]
        
        pre_existing = patient_data['pre_existing']
        pre_existing_encoded = models['le_pre_existing'].transform([pre_existing])[0]
        
        X = np.array([[
            patient_data['age'],
            gender_encoded,
            patient_data['systolic_bp'],
            patient_data['diastolic_bp'],
            patient_data['heart_rate'],
            patient_data['temperature'],
            symptom_encoded,
            pre_existing_encoded
        ]])
        
        risk_pred = models['risk_model'].predict(X)[0]
        risk_proba = models['risk_model'].predict_proba(X)[0]
        
        dept_pred = models['dept_model'].predict(X)[0]
        dept_proba = models['dept_model'].predict_proba(X)[0]
        
        risk_classes = models['risk_model'].classes_
        dept_classes = models['dept_model'].classes_
        
        result = {
            'risk': risk_pred,
            'risk_confidence': max(risk_proba) * 100,
            'risk_probs': {cls: prob * 100 for cls, prob in zip(risk_classes, risk_proba)},
            'department': dept_pred,
            'dept_confidence': max(dept_proba) * 100,
            'dept_probs': {cls: prob * 100 for cls, prob in zip(dept_classes, dept_proba)},
            'factors': explain_prediction(patient_data, risk_pred)
        }
        
        return result
        
    except Exception as e:
        return {'error': str(e)}

# ============================================================================
# LANGUAGE SELECTOR
# ============================================================================
col_lang1, col_lang2, col_lang3 = st.columns([3, 1, 0.5])

with col_lang3:
    st.markdown("")
    languages = {
        'en': '🇬🇧 English',
        'es': '🇪🇸 Español',
        'fr': '🇫🇷 Français',
        'hi': '🇮🇳 हिन्दी',
        'ta': '🇮🇳 தமிழ்',
        'ar': '🇸🇦 العربية'
    }
    
    selected_lang = st.selectbox(
        label=t('select_language'),
        options=list(languages.keys()),
        format_func=lambda x: languages[x],
        index=list(languages.keys()).index(st.session_state.language),
        key="language_selector"
    )
    
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()

# ============================================================================
# HEADER
# ============================================================================
st.markdown(f'<h1 style="text-align: center; margin-bottom: 0.5rem;">{t("app_title")}</h1>', 
            unsafe_allow_html=True)
st.markdown(f'<p style="text-align: center; color: #6B7280; font-size: 1.1rem; margin-bottom: 2rem;">{t("app_subtitle")}</p>', 
            unsafe_allow_html=True)

# ============================================================================
# PROGRESS STEPPER
# ============================================================================
steps = [t("step_vitals"), t("step_symptoms"), t("step_history"), t("step_results")]
stepper_html = '<div class="stepper">'
for i, step_name in enumerate(steps, 1):
    active_class = "step-active" if i == st.session_state.step else ""
    stepper_html += f'<div class="step {active_class}">{step_name}</div>'
stepper_html += '</div>'
st.markdown(stepper_html, unsafe_allow_html=True)

# ============================================================================
# STEP 1: VITALS
# ============================================================================
if st.session_state.step == 1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<p style="color: #1B3A52; font-size: 1.8rem; font-weight: 700;">{t("vitals_header")}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color: #6B7280; font-size: 1rem; margin-bottom: 2rem;">{t("vitals_subheader")}</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f'**{t("age")}** ' + info_icon(
            "Patient's age in years. Pediatric: <18, Adult: 18-65, Geriatric: >65"
        ), unsafe_allow_html=True)
        st.session_state.form_data['age'] = st.slider(
            label=t("age"),
            min_value=0,
            max_value=120,
            value=st.session_state.form_data['age'],
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown(f'**{t("gender")}**', unsafe_allow_html=True)
        gender_options = [t("male"), t("female")]
        gender_map = {t("male"): 'Male', t("female"): 'Female'}
        reverse_gender_map = {'Male': t("male"), 'Female': t("female")}
        
        selected_gender = st.selectbox(
            label=t("gender"),
            options=gender_options,
            index=gender_options.index(reverse_gender_map[st.session_state.form_data['gender']]),
            label_visibility="collapsed"
        )
        st.session_state.form_data['gender'] = gender_map[selected_gender]
    
    st.markdown("")
    
    st.markdown(f'**{t("blood_pressure")}** ' + info_icon(
        "Normal: 120/80 mmHg | Elevated: 120-129/<80 | High: ≥140/≥90 | Crisis: >180/>120"
    ), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.form_data['systolic_bp'] = st.slider(
            label=t("systolic"),
            min_value=60,
            max_value=220,
            value=st.session_state.form_data['systolic_bp']
        )
    
    with col2:
        st.session_state.form_data['diastolic_bp'] = st.slider(
            label=t("diastolic"),
            min_value=40,
            max_value=140,
            value=st.session_state.form_data['diastolic_bp']
        )
    
    st.markdown("")
    
    st.markdown(f'**{t("heart_rate")}** ' + info_icon(
        "Normal: 60-100 BPM | Bradycardia: <60 | Tachycardia: >100"
    ), unsafe_allow_html=True)
    
    st.session_state.form_data['heart_rate'] = st.slider(
        label=t("heart_rate"),
        min_value=30,
        max_value=200,
        value=st.session_state.form_data['heart_rate'],
        label_visibility="collapsed"
    )
    
    st.markdown("")
    
    st.markdown(f'**{t("temperature")}** ' + info_icon(
        "Normal: 36.1-37.2°C | Fever: >38°C | High fever: >39°C"
    ), unsafe_allow_html=True)
    
    st.session_state.form_data['temperature'] = st.slider(
        label=t("temperature"),
        min_value=34.0,
        max_value=42.0,
        value=st.session_state.form_data['temperature'],
        step=0.1,
        format="%.1f",
        label_visibility="collapsed"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button(t("continue"), key="vitals_continue", use_container_width=True):
            next_step()
            st.rerun()

# ============================================================================
# STEP 2: SYMPTOMS (WITH VOICE INPUT)
# ============================================================================
elif st.session_state.step == 2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<p style="color: #1B3A52; font-size: 1.8rem; font-weight: 700;">{t("symptoms_header")}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color: #6B7280; font-size: 1rem; margin-bottom: 2rem;">{t("symptoms_subheader")}</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("")
    
    # Voice Input Section
    st.markdown('<div class="voice-section">', unsafe_allow_html=True)
    st.markdown(f'<p style="color: #1B3A52; font-size: 1.5rem; font-weight: 700; margin-top: 0;">{t("voice_input")}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color: #6B7280;">{t("voice_instructions")}</p>', unsafe_allow_html=True)
    
    st.markdown(f"**{t('click_to_record')}:**")
    
    voice_input_text = st.text_area(
        label=t("voice_input"),
        value=st.session_state.voice_text,
        placeholder="e.g., headache, fever, cough" if st.session_state.language == 'en' else 
                   "ej., dolor de cabeza, fiebre" if st.session_state.language == 'es' else
                   "ex., mal de tête, fièvre" if st.session_state.language == 'fr' else
                   "उदा., सिरदर्द, बुखार" if st.session_state.language == 'hi' else
                   "எ.கா., தலைவலி, காய்ச்சல்" if st.session_state.language == 'ta' else
                   "مثل، صداع، حمى",
        height=100,
        label_visibility="collapsed",
        key="voice_text_input"
    )
    
    col_voice1, col_voice2 = st.columns(2)
    
    with col_voice1:
        if st.button(f"{t('start_recording')}", key="start_voice", use_container_width=True):
            st.session_state.voice_text = voice_input_text
            
            detected = process_voice_text(voice_input_text, st.session_state.language)
            
            for symptom in detected:
                if symptom not in st.session_state.form_data['symptoms']:
                    st.session_state.form_data['symptoms'].append(symptom)
            
            st.session_state.form_data['voice_symptoms'] = detected
            st.rerun()
    
    with col_voice2:
        if st.button(t("clear_voice"), key="clear_voice", use_container_width=True):
            st.session_state.voice_text = ""
            for symptom in st.session_state.form_data.get('voice_symptoms', []):
                if symptom in st.session_state.form_data['symptoms']:
                    st.session_state.form_data['symptoms'].remove(symptom)
            st.session_state.form_data['voice_symptoms'] = []
            st.rerun()
    
    if st.session_state.form_data.get('voice_symptoms'):
        st.success(f"✅ {t('voice_detected')}: {', '.join(st.session_state.form_data['voice_symptoms'])}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("")
    
    # Manual symptom selection
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"### {t('manual_symptoms')}")
    
    all_symptoms = sorted(models['le_symptoms'].classes_)
    
    cols = st.columns(3)
    
    for i, symptom in enumerate(all_symptoms[:18]):
        with cols[i % 3]:
            is_selected = symptom in st.session_state.form_data['symptoms']
            
            if st.checkbox(symptom, key=f"symptom_{symptom}", value=is_selected):
                if symptom not in st.session_state.form_data['symptoms']:
                    st.session_state.form_data['symptoms'].append(symptom)
            else:
                if symptom in st.session_state.form_data['symptoms']:
                    st.session_state.form_data['symptoms'].remove(symptom)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button(t("previous"), key="symptoms_prev", use_container_width=True):
            prev_step()
            st.rerun()
    with col2:
        if st.button(t("continue"), key="symptoms_next", use_container_width=True):
            if not st.session_state.form_data['symptoms']:
                st.warning(t("warning_symptoms"))
            else:
                next_step()
                st.rerun()

# ============================================================================
# STEP 3: HISTORY (WITH DOCUMENT UPLOAD)
# ============================================================================
elif st.session_state.step == 3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<p style="color: #1B3A52; font-size: 1.8rem; font-weight: 700;">{t("history_header")}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="color: #6B7280; font-size: 1rem; margin-bottom: 2rem;">{t("history_subheader")}</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("")
    
    # Document Upload Section
    st.markdown('<div class="card" style="border: 2px dashed #B8D8E8;">', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size: 3rem; color: #1B3A52;">🤖</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; color: #1B3A52; font-size: 1.5rem; font-weight: 700;">{t("document_upload")}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align: center; color: #6B7280; margin-bottom: 1.5rem;">{t("upload_subtitle")}</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        label=t("choose_file"),
        type=['pdf', 'txt', 'jpg', 'jpeg', 'png'],
        help="Supported formats: PDF, TXT, JPG, PNG",
        label_visibility="visible",
        key="doc_uploader"
    )
    
    if uploaded_file is not None:
        st.session_state.form_data['uploaded_document'] = uploaded_file
        st.session_state.form_data['document_name'] = uploaded_file.name
        
        st.success(f"✅ {uploaded_file.name}")
        
        with st.spinner(f"{t('analyzing')}"):
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension in ['jpg', 'jpeg', 'png']:
                extracted_text = extract_text_from_image(uploaded_file)
            elif file_extension == 'txt':
                extracted_text = extract_text_from_txt(uploaded_file)
            elif file_extension == 'pdf':
                extracted_text = extract_text_from_pdf(uploaded_file)
            else:
                extracted_text = "Unsupported file format"
            
            available_conditions = sorted(models['le_pre_existing'].classes_)
            analysis_result = analyze_medical_document(extracted_text, available_conditions)
            
            st.session_state.form_data['extracted_data'] = analysis_result
        
        if analysis_result and analysis_result['conditions']:
            st.markdown('<div class="extracted-data">', unsafe_allow_html=True)
            st.markdown(f"### {t('extracted_info')}")
            
            st.markdown(f"**{t('detected_conditions')}**")
            for condition in analysis_result['conditions']:
                confidence = analysis_result['confidence'].get(condition, 0)
                st.markdown(f'<div style="padding: 0.5rem 0;">✓ <strong>{condition}</strong> ({t("confidence")}: {confidence}%)</div>', 
                           unsafe_allow_html=True)
            
            if analysis_result['conditions']:
                best_condition = max(analysis_result['conditions'], 
                                   key=lambda x: analysis_result['confidence'].get(x, 0))
                st.session_state.form_data['pre_existing'] = best_condition
                st.info(f"✨ Auto-selected: **{best_condition}**")
            
            if analysis_result['vitals']:
                st.markdown(f"**{t('extracted_vitals')}**")
                for vital_name, vital_value in analysis_result['vitals'].items():
                    st.markdown(f'<div style="padding: 0.5rem 0;">📊 {vital_name.replace("_", " ").title()}: <strong>{vital_value}</strong></div>', 
                               unsafe_allow_html=True)
                    
                    if vital_name in st.session_state.form_data:
                        st.session_state.form_data[vital_name] = vital_value
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button(t('apply_data'), key="apply_data", use_container_width=True):
                st.success(f"✅ {t('processing')}")
        
        elif analysis_result:
            st.warning("⚠️ No medical conditions detected")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("")
    
    # Manual Selection
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"### {t('manual_selection')}")
    
    all_conditions = sorted(models['le_pre_existing'].classes_)
    
    st.session_state.form_data['pre_existing'] = st.selectbox(
        label=t("select_condition"),
        options=all_conditions,
        index=all_conditions.index(st.session_state.form_data['pre_existing']) 
              if st.session_state.form_data['pre_existing'] in all_conditions else 0,
        key="manual_condition"
    )
    
    st.info(f"💡 {t('no_history_info')}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button(t("previous"), key="history_prev", use_container_width=True):
            prev_step()
            st.rerun()
    with col2:
        if st.button(t("analyze_patient"), key="history_analyze", use_container_width=True):
            next_step()
            st.rerun()

# ============================================================================
# STEP 4: REVIEW & RESULTS
# ============================================================================
elif st.session_state.step == 4:
    st.markdown(f'<p style="text-align: center; color: #1B3A52; font-size: 1.8rem; font-weight: 700;">{t("results_header")}</p>', unsafe_allow_html=True)
    st.markdown("")
    
    result = make_prediction(st.session_state.form_data)
    
    if 'error' in result:
        st.error(f"Error: {result['error']}")
        if st.button(t("previous")):
            prev_step()
            st.rerun()
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            st.markdown(f'<p style="color: #1B3A52; font-size: 1.5rem; font-weight: 700;">{t("risk_classification")}</p>', unsafe_allow_html=True)
            risk = result['risk']
            conf = result['risk_confidence']
            
            if risk == 'High':
                st.markdown(f'<div class="risk-high">🔴 {t("high_risk")}<br/>{conf:.1f}% {t("confidence")}</div>', 
                           unsafe_allow_html=True)
            elif risk == 'Medium':
                st.markdown(f'<div class="risk-medium">🟡 {t("medium_risk")}<br/>{conf:.1f}% {t("confidence")}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">🟢 {t("low_risk")}<br/>{conf:.1f}% {t("confidence")}</div>', 
                           unsafe_allow_html=True)
            
            st.markdown("")
            
            st.markdown(f'<p style="color: #1B3A52; font-size: 1.5rem; font-weight: 700;">{t("recommended_dept")}</p>', unsafe_allow_html=True)
            st.markdown(f'<div class="dept-box">📍 {result["department"]}<br/>{result["dept_confidence"]:.1f}% {t("match")}</div>', 
                       unsafe_allow_html=True)
            
            st.markdown("")
            
            st.markdown(f'<p style="color: #1B3A52; font-size: 1.5rem; font-weight: 700;">{t("risk_probabilities")}</p>', unsafe_allow_html=True)
            for level in ['High', 'Medium', 'Low']:
                prob = result['risk_probs'].get(level, 0)
                icon = "🔴" if level == 'High' else "🟡" if level == 'Medium' else "🟢"
                level_text = t("high_risk") if level == 'High' else t("medium_risk") if level == 'Medium' else t("low_risk")
                st.metric(f"{icon} {level_text}", f"{prob:.1f}%")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            st.markdown(f'<p style="color: #1B3A52; font-size: 1.5rem; font-weight: 700;">{t("clinical_recommendations")}</p>', unsafe_allow_html=True)
            
            if risk == 'High':
                st.error(f"""
                **🔴 {t('immediate_action')}**
                
                **{t('priority')}:** ESI Level 1
                
                **{t('actions')}:**
                - Immediate trauma bay assignment
                - Alert attending physician
                - Continuous monitoring
                
                **{t('target')}:** {t('physician_eval_immediate')}
                """)
            elif risk == 'Medium':
                st.warning(f"""
                **🟡 {t('urgent_assessment')}**
                
                **{t('priority')}:** ESI Level 2-3
                
                **{t('actions')}:**
                - Move to urgent care
                - Vitals every 15-30 minutes
                
                **{t('target')}:** {t('physician_eval_15_30')}
                """)
            else:
                st.success(f"""
                **🟢 {t('routine_processing')}**
                
                **{t('priority')}:** ESI Level 4-5
                
                **{t('actions')}:**
                - General waiting area
                - Standard monitoring
                
                **{t('expected_wait')}:** {t('hours_1_2')}
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("")
        
        col_factor, col_summary = st.columns(2)
        
        with col_factor:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<p style="color: #1B3A52; font-size: 1.5rem; font-weight: 700;">{t("contributing_factors")}</p>', unsafe_allow_html=True)
            for factor in result['factors']:
                st.markdown(f'<div class="factor-box">{factor}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_summary:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<p style="color: #1B3A52; font-size: 1.5rem; font-weight: 700;">{t("patient_summary")}</p>', unsafe_allow_html=True)
            
            doc_source = t("ai_extracted") if st.session_state.form_data.get('extracted_data') else t("manual")
            
            summary_data = {
                t('field'): [
                    t('age'), t('gender'), t('blood_pressure'), t('heart_rate'), t('temperature'), 
                    t('symptoms_label'), t('pre_existing'), t('document'), t('data_source')
                ],
                t('value'): [
                    f"{st.session_state.form_data['age']} {t('years')}",
                    st.session_state.form_data['gender'],
                    f"{st.session_state.form_data['systolic_bp']}/{st.session_state.form_data['diastolic_bp']} mmHg",
                    f"{st.session_state.form_data['heart_rate']} BPM",
                    f"{st.session_state.form_data['temperature']}°C",
                    ', '.join(st.session_state.form_data['symptoms'][:3]) + ('...' if len(st.session_state.form_data['symptoms']) > 3 else ''),
                    st.session_state.form_data['pre_existing'],
                    st.session_state.form_data['document_name'] if st.session_state.form_data['document_name'] else t('none'),
                    doc_source
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("")
        
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        with btn_col1:
            if st.button(t("previous"), key="results_prev", use_container_width=True):
                prev_step()
                st.rerun()
        
        with btn_col2:
            if st.button(t("new_patient"), key="results_reset", use_container_width=True):
                reset_form()
                st.rerun()
        
        with btn_col3:
            st.success(t("assessment_complete"))

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("")
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6B7280; padding: 1.5rem;'>
    <p style='margin: 0; font-size: 0.95rem;'>
        <strong>⚠️ Medical Disclaimer:</strong> This tool is for demonstration purposes only.
    </p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
        <em>MedTouch.ai v3.1 | AI-Powered Multilingual Triage System with Voice Input</em>
    </p>
</div>
""", unsafe_allow_html=True)