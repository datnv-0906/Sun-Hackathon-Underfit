from django.views.generic import TemplateView
from braces.views import JSONResponseMixin, AjaxResponseMixin
from PIL import Image
import numpy as np
import cv2 as cv

from common_service.service import text_detection_service, get_text_service


class Conflict(TemplateView):
    template_name = "front-end/conflict.html"


class Sidefx(TemplateView):
    template_name = "front-end/sidefx.html"


class IndexView(JSONResponseMixin, AjaxResponseMixin, TemplateView):
    template_name = "front-end/index.html"

    def post_ajax(self, request, *args, **kwargs):
        img_temp = request.FILES["files"]
        img = Image.open(img_temp)
        if not img.format == 'RGB':
            img = img.convert('RGB')

        img = np.asarray(img)
        area_crops = text_detection_service([img])
        text_recogs = []
        for i in range(len(area_crops)):
            text_recogs.append(get_text_service(np.array(area_crops[i])))

        return self.render_json_response({"text_recogs": text_recogs}, 200)

    def get_context_data(self, **kwargs):
        context = super(IndexView, self).get_context_data(**kwargs)
        return context
