from django.urls import path
from .views import PreguntaResponder, test_view

urlpatterns = [
    path('responder/', PreguntaResponder.as_view(), name='responder'),
    path('test/', test_view, name='test-view'),
]
