from django.db import models

class College(models.Model):
    name = models.CharField(max_length=255, unique=True)
    stream = models.CharField(max_length=255)
    percentage = models.FloatField()

    def __str__(self):
        return self.name

class Company(models.Model):
    name = models.CharField(max_length=255, unique=True)
    region = models.CharField(max_length=255)

    def __str__(self):
        return self.name

class Placement(models.Model):
    company = models.ForeignKey(Company, on_delete=models.CASCADE)
    college = models.ForeignKey(College, on_delete=models.CASCADE)
    salary = models.FloatField()
    year = models.IntegerField()

    def __str__(self):
        return f"{self.company.name} - {self.college.name} ({self.year})"
