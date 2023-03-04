To make EventList.read work on Fermi/GBM data, you will need to follow these steps:

Install the HEASOFT software package: HEASOFT is a collection of software tools developed by NASA for analyzing astrophysical data. It includes the FTOOLS package, which contains the EventList.read function that can be used to read in Fermi/GBM data. HEASOFT can be downloaded from the NASA website at https://heasarc.gsfc.nasa.gov/lheasoft/download.html.

Download the Fermi/GBM data: The data can be downloaded from the Fermi Science Support Center (FSSC) website at https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3table.pl?tablehead=name%3Dfermigbmcat&Action=More+Options. You will need to select the time range and the energy range for the data you want to download.

Convert the data into the FITS format: The Fermi/GBM data is provided in the GBM format, which is not compatible with EventList.read. You will need to convert the data into the FITS format using the Fermi Science Tools (FST). FST is included in the HEASOFT software package.

Use EventList.read to read in the data: Once the data has been converted into the FITS format, you can use EventList.read to read in the data. Here is an example code snippet:


```
from astropy.io import fits
from astropy.table import Table

filename = 'gbm_data.fits'
hdulist = fits.open(filename)
tbdata = Table(hdulist[1].data)
events = tbdata['TIME']
```


In this example, 'gbm_data.fits' is the name of the FITS file containing the Fermi/GBM data. The data is read into a Table object using the fits.open function, and the events are extracted from the table using the 'TIME' column.
