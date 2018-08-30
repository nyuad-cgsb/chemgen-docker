# Use an rocker as a parent image
FROM r-base

# Copy the current directory contents into the container at /app
ADD . /Users/alan/projects/gunsiano/dockers/michelle-pipeline

# Set the working directory
WORKDIR /Users/alan/projects/gunsiano/dockers/michelle-pipeline

# Make port 80 available to the world outside this container
# EXPOSE 80

# Define environment variable
# ENV NAME World

RUN apt-get update && apt-get install -y build-essential
RUN Rscript -e 'install.packages(c("plyr","dplyr","tidyr","ggplot2"))'

# Run app.py when the container launches
CMD ["conda install -y r-dplyr r-plyr r-ggplot2 r-tidyr"]
CMD ["Rscript", "enh_for_Jillian.R"]
