from django.shortcuts import render, redirect
from django.conf import settings
from .forms import UploadForm
from .inference import run_detection_and_segmentation
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ultralytics")


def upload_image(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            upload_instance = form.save()
            uploaded_path = upload_instance.image.path
            output_dir = os.path.join(settings.MEDIA_ROOT, 'outputs')

            # Run detection + segmentation
            detection_file, segmentation_file, tumor_summary = run_detection_and_segmentation(
                uploaded_path, output_dir, device='cpu'
            )

            return render(request, 'segmentation/result.html', {
                'original_url': os.path.join(settings.MEDIA_URL, 'uploads', os.path.basename(uploaded_path)),
                'detection_url': os.path.join(settings.MEDIA_URL, 'outputs', detection_file),
                'segmentation_url': os.path.join(settings.MEDIA_URL, 'outputs', segmentation_file),
                'tumor_summary': tumor_summary
            })
    else:
        form = UploadForm()
    return render(request, 'segmentation/upload.html', {'form': form})


def view_result(request, output_filename):
    out_url = os.path.join(settings.MEDIA_URL, 'outputs', output_filename)
    return render(request, 'segmentation/result.html', {'out_url': out_url})

