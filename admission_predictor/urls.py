from django.contrib import admin
from django.urls import path
from main_app.views import predict_company, predict_colleges, index

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', index, name='index'),
    path('predict_company/', predict_company, name='predict_company'),
    path('predict_colleges/', predict_colleges, name='predict_colleges'),
]