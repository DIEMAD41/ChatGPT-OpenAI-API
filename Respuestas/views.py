import json

import numpy as np
import openai
from django.conf import settings
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import PreguntaFrecuente
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity


class PreguntaResponder(APIView):

    def post(self, request, *args, **kwargs):
        datos = request.data
        pregunta = datos.get('questions')[0].get('text') if datos.get('questions') else None
        especificaciones = self.obtener_especificaciones(datos)

        if not pregunta:
            return Response({"error": "No se proporcionó una pregunta"}, status=status.HTTP_400_BAD_REQUEST)

        # Buscar en el vector store
        faq, similarity = self.buscar_pregunta_similar(pregunta)
        if similarity > 0.55:  # Umbral de similitud
            respuesta = faq.respuesta
        else:
            # Si no se encuentra, usa la API de OpenAI (ChatGPT)
            respuesta = self.obtener_respuesta_de_chatgpt(pregunta, especificaciones)

            if respuesta:
                self.indexar_pregunta(pregunta, respuesta)
            else:
                return Response({"error": "No se pudo obtener una respuesta"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Modificar el JSON de salida
        datos_modificados = self.modificar_json(datos, respuesta)
        return JsonResponse(datos_modificados)

    def obtener_especificaciones(self, datos):
        especificaciones = {
            "id_item": datos.get('id_item'),
            "title": datos.get('title'),
            "price": datos.get('price'),
            "currency_id": datos.get('currency_id'),
            "available_quantity": datos.get('available_quantity'),
            "sold_quantity": datos.get('sold_quantity'),
            "condition": datos.get('condition'),
            "attributes": datos.get('attributes'),
            "warranty": datos.get('warranty'),
        }
        return especificaciones

    def obtener_respuesta_de_chatgpt(self, pregunta, especificaciones):
        client = OpenAI(api_key=settings.OPENAI_API_KEY)  # Asegúrate de que la API Key esté configurada
        prompt = f"Especificaciones del producto:\n{especificaciones}\n\nPregunta: {pregunta}\n\nResponde de acuerdo con las especificaciones del producto:"
        MODEL = "gpt-3.5-turbo"

        messages = [
            {"role": "system",
             "content": "Eres un vendedor que responde mensajes relacionados a la venta de productos en linea. No hace falta que menciones que fue segun las especificaciones proporcionadas"},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=250,
            temperature=0.2,
        )

        # Acceder al contenido de la respuesta
        answer = response.choices[0].message.content
        return answer

    def crear_embedding(self, texto):
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        response = client.embeddings.create(
            input=texto,
            model="text-embedding-3-small"
        )

        embedding = response.data[0].embedding
        return embedding

    def indexar_pregunta(self, pregunta, respuesta):
        embedding = self.crear_embedding(pregunta)
        PreguntaFrecuente.objects.create(pregunta=pregunta, respuesta=respuesta, embedding=json.dumps(embedding))

    def buscar_pregunta_similar(self, pregunta):
        embedding = self.crear_embedding(pregunta)
        faqs = PreguntaFrecuente.objects.all()
        if not faqs:
            return None, 0

        max_similarity = 0
        best_faq = None
        for faq in faqs:
            faq_embedding = json.loads(faq.embedding)
            similarity = cosine_similarity([embedding], [faq_embedding])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                best_faq = faq

        return best_faq, max_similarity

    def modificar_json(self, datos, respuesta):
        if 'questions' in datos and len(datos['questions']) > 0:
            datos['questions'][0]['answer'] = respuesta
            datos['questions'][0]['status'] = 'answered'
        return datos


def test_view(request):
    return JsonResponse({"message": "Test view is working!"})