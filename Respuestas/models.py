from django.db import models

# Create your models here.

class PreguntaFrecuente(models.Model):
    pregunta = models.TextField()
    respuesta = models.TextField()
    embedding = models.TextField(null=True, blank=True)
    fecha_creacion = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.pregunta