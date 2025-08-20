from django.db import models

# Create your models here.
class ckdModel(models.Model):

    FileAccessed_or_not=models.IntegerField()
    FileModified_or_not=models.IntegerField()
    FileUploaded_or_not=models.IntegerField()
    FileDownloaded_or_not=models.IntegerField()
    MoveToDeletedItems=models.IntegerField()
    IsRiskyHour_or_not=models.IntegerField()
    UserLoginFailed_or_not=models.IntegerField()
    GeoLocation_IND_or_not=models.IntegerField()
    Unknow_ResultStatus_or_not=models.IntegerField()
