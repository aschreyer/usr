# USR: Ultrafast Shape Recognition
- - - 


Project description

## Installation

**USR** can simply be installed with `pip` from the Exscientia PyPI server using

    $ pip install rd-usr

which will install the latest stable release. For other versions, the version can be 
explicitly specified using the  normal pip syntax, e.g. `pip install exs-usr==1.0`. 

### Developer installation

For developers who directly work on the source code, it is recommended to install 
**USR** in development mode. For that, please clone the repository first with

    $ git clone git@bitbucket.org:exscientia/usr.git

and then install the repo in `pip` development mode:

    $ pip install --extra-index-url https://pypi.anaconda.org/OpenEye/simple -e .[tests]

This step will also install all the dependencies required for running the test suite.

### Using Docker

As as alternative, The **USR** Docker container image can be used to run **USR** without 
installing any software. The [Docker](https://www.docker.com) software can be downloaded 
for desktop systems (e.g. MacOS) at https://www.docker.com/get-started.

First, log into the Amazon ECR service with docker login:
    
    $ eval $(aws ecr get-login --no-include-email --region eu-west-1)

Or on Windows:

    $ Invoke-Expression -Command (aws ecr get-login --no-include-email)

The **USR** docker image can the be pulled from the Amazon ECR repository with:

    $ docker pull 752357919085.dkr.ecr.eu-west-1.amazonaws.com/usr[:TAG]
    
Where tag can be `latest`, which pulls the latest image (not recommended), the name of a 
branch (e.g. `usr:master`), or a version number, for example `usr:1.0.0`. In the 
following, the `master` branch will be used as example.
    
The following step is not strictly necessary but for convenience the image can be tagged 
with a simpler name:
    
    $ docker tag 752357919085.dkr.ecr.eu-west-1.amazonaws.com/usr:master usr:master

It is possible to use **USR** from within the container interactively but the simpler 
method is to treat the container as a single binary and run the `usr` CLI application 
directly from the host:

    $ docker run --volume=[HOST PATH]:/[CONTAINER PATH] --workdir="[CONTAINER PATH]" usr:master [APP] [OPTIONS]

The `--volume` option can be used to make a directory on the host available on in the 
container. With this approach it is possible to read a start population directly from 
the host and write the evolution results there directly as well.

### Software requirements

**USR** currently requires at least Python 3.6 and the OpenEye toolkit distribution.

The `pip` installation automatically installs all required dependencies. However, the 
full list of requirements that are typically installed can be seen in the 
`requirements.txt` file.

### Testing

**USR** comes with a comprehensive test suite that can be started by calling `$ py.test` 
in the source directory.
