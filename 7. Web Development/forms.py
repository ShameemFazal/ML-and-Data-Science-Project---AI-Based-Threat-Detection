from django import forms
from .models import *


class ckdForm(forms.ModelForm):
    class Meta():
        model=ckdModel
        fields=['FileAccessed_or_not','FileModified_or_not','FileUploaded_or_not',
        'FileDownloaded_or_not','MoveToDeletedItems','IsRiskyHour_or_not',
        'UserLoginFailed_or_not','GeoLocation_IND_or_not','Unknow_ResultStatus_or_not']
