# CONFIGURATION DIRECTIVES
# Compilers 
CC = g++
# the build target executable:
TARGET = binary_validator
# FLAGS 
# CC compiler flags:
CFLAGS  = -g

# Littelendian and Bigendian flags, by default if value is not set is Littelendian if value is set to -DBIGENDIAN is Bigendian
# -DBIGENDIAN
ENDIANFLAGS = 
# OUTPUT FOLDER
OUTPUTFOLDER = ./bin/

all:
	mkdir -p $(OUTPUTFOLDER)
	$(CC) $(CFLAGS) $(ENDIANFLAGS) -o $(OUTPUTFOLDER)$(TARGET) binary_validator.c


clean:
	rm -rf $(OUTPUTFOLDER)$(TARGET)