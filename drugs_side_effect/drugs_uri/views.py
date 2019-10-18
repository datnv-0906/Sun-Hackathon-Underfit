from django.views.generic import TemplateView
from django.http import StreamingHttpResponse
from braces.views import JSONResponseMixin, AjaxResponseMixin


class IndexView(TemplateView, JSONResponseMixin, AjaxResponseMixin):
    template_name = "front-end/index.html"

    def get_context_data(self, **kwargs):
        context = super(IndexView, self).get_context_data(**kwargs)
        return context