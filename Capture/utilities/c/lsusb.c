#include <stdio.h>
#include <sys/types.h>

#include <libusb.h>




int open(struct libusb_device_handle **handle){
	/*
	opens device with signature 0x20b1,0x0101
	claims interface 0
	writes dev handle as adress to pointer it recieves a pointer to from python
	*/
	
	//initialize Libusb
	int err = libusb_init(NULL);
	if (err < 0){
		fprintf(stderr, "ERROR: failed to init libusb");
		return -1;
	}
	libusb_set_debug(NULL, 0);
	//open device find by pid/vid signature
	struct libusb_device_handle *dev_handle = NULL;
	dev_handle = libusb_open_device_with_vid_pid(NULL,0x20b1,0x0101);  //0101 cam //0401 vcom
	if (dev_handle == NULL) { 
	 	printf("ERROR: Couldn't find device\n"); 
	    return -1; 
		} 
	printf("USB Camera Interface found. Opening Device \n");
	int bConfigurationValue;
	err = libusb_get_configuration(dev_handle, &bConfigurationValue);
	if(err != 0){
		printf("Could not get bConfigurationValue");
		return -1;
		}
	printf("bConfigurationValue: %d \n", bConfigurationValue);

	err = libusb_set_configuration(dev_handle, bConfigurationValue);
	if(err != 0){
		printf("Could not set bConfigurationValue");
		return -1;
		}
	printf("bConfigurationValue set\n");
	

	//claim interface 0
	err = libusb_claim_interface(dev_handle,0);
	if(err != 0){
		printf("Could not claim interface");
		return -1;
		}
	
	//passing the handle to python
	*handle = dev_handle; 
	printf("device set handle adress: %p \n", *handle);
	
	return 0;
}


int read(struct libusb_device_handle **handle,unsigned char  *buffer,int len, char ep){
	/*
	read data, 

	*/
	//printf("device to read from: %p \n", *handle);

	int transferred, err;
	err = libusb_bulk_transfer(*handle, ep , buffer, len, &transferred, 10);
	if(err != 0 ){
		printf("ERROR: reading  read %i ,error %i \n", transferred, err);
		return -1;
		}
	if((transferred != len)){
		printf("ERROR: reading  data  incomplete read %i ,error %i \n", transferred, err);
		return -1;
		}
	return 0;
}




int write(struct libusb_device_handle **handle,unsigned char  *buffer,int len, char ep){
	/*
	write
	*/
	//printf("device to read from: %p \n", *handle);

	int transferred, err;

	err = libusb_bulk_transfer(*handle, ep , buffer, len, &transferred, 1);
	if(err != 0 || (transferred != len) ){
		printf("ERROR: data could not be written or incomplete  transferred %i ,error %i \n", transferred, err);
		return -1;
		}

	return 0;
}
 
int close(struct libusb_device_handle **handle){
	/*
	cleaning up
	*/
	printf("releasing interface 0 \n");
	libusb_release_interface(*handle,0);
	printf("closing device with handle adress %p \n", *handle);
	libusb_close(*handle);
	printf("DONE: close dev_handle \n");
	libusb_exit(NULL);
	return 0;
}



//debugging fns

int list(void)
{
	/*
	lists devices and their desciptors
	*/
	int err = libusb_init(NULL);
	if (err < 0){
		fprintf(stderr, "ERROR: failed to init libusb");
		return -1;
	}

	// discover devices
	libusb_device **list;
	ssize_t cnt = libusb_get_device_list(NULL, &list);
	ssize_t i = 0;
	
	if (cnt < 0){
		fprintf(stderr, "ERROR: No libUSB device list");
		return -1;
	}
	for (i = 0; i < cnt; i++) { //interate through usb devices
	    libusb_device *device = list[i];
	    
		struct libusb_device_descriptor desc;
		int err = libusb_get_device_descriptor(device, &desc);
		if (err < 0) {
			fprintf(stderr, "failed to get device descriptor");
			return -1;
		}

		//get configuration decriptor for configuration 0
		struct libusb_config_descriptor *conf;
		err = libusb_get_config_descriptor(device,(uint8_t) 0,  &conf);
		if (err < 0) {
			fprintf(stderr, "failed to get config descriptor");
			return -1;
		}


		printf("MaxPower: %d  \n", (*conf).MaxPower);
		printf("NumInterfaces: %d  \n", (*conf).bNumInterfaces);
		struct libusb_interface interface[(*conf).bNumInterfaces]; // array containing the interfaces 
		struct libusb_interface_descriptor int_des; // interface descriptor
		int j;
		for ( j = 0; j < (*conf).bNumInterfaces; j++){
			interface[j] = conf[0].interface[j];
			int_des = *interface[j].altsetting;
			printf("Interface %i  with  %d endpoints \n",j, int_des.bNumEndpoints);
		}

		//print decriptor information
		printf("Device: %04x:%04x bus %d, device %d,Class %02x, numConfigurations %d, \n",
			desc.idVendor, desc.idProduct,
			libusb_get_bus_number(device), 
			libusb_get_device_address(device), 
			desc.bDeviceClass,
			desc.bNumConfigurations);

		libusb_free_config_descriptor (conf);
		}
	libusb_free_device_list(list, 1);
	libusb_exit(NULL);
	return 0;
}

int test(void)
{
	printf("TEST");
	return 0;
}
