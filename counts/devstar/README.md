## Docker Image for Devstar

### Get devstar
Devstar is in a private repo :https://github.com/GunsalusPiano/devstar
Download as a zip, and put in this ('counts/devstar') folder in order to run it.

### Build the Image

```
docker build -t devstar .
```

### Run the Image In Interactive Mode
```
docker run -it -v `pwd`/example:/example devstar
```

### Run the image with a batch job configuration

```
docker run -v `pwd`/example:/example devstar /scripts/devstar-master/devstar analyze -C /path/to/config
```

Ensure that the outpath the devstar config exists and is writable before running
