from django.urls import path
from drugs_uri.views import IndexView


urlpatterns = [
    path('', IndexView.as_view(), name="index"),
]
