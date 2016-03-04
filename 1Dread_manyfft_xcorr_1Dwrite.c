
#include "stdlib.h"
#include "stdbool.h"
#include <complex.h> 
#include "hdf5.h"
#include <mpi.h>
#include <fftw3-mpi.h>

#define H5FILE_NAME     "testfile.h5"
#define DATASETNAME 	"X_Dataset" 
#define NX     1                     /* dataset dimensions */
#define NY     1000
#define RANK   2
#define MAX_NAME 20

int
main (int argc, char **argv)
{
    /*
     * HDF5 APIs definitions
     */ 	
    hid_t       file_id; 
    hid_t       dataset_id;               /* file and dataset identifiers */
    hid_t       filespace, memspace;      /* file and memory dataspace identifiers */
    hsize_t     dimsf[RANK];                 /* dataset dimensions */ 
    hsize_t     tot_proc_datasets[RANK];	         
    hsize_t	count[RANK];                 /* hyperslab selection parameters */
    hsize_t	offset[RANK];
    hsize_t     stride[RANK];
    hsize_t     block[RANK];
    hid_t	plist_id;                 /* property list identifier */
    int         i;
    herr_t	status;
    hssize_t 	data_dim;
    size_t      size; 
    hid_t       datatype;
    int         rank;
    hsize_t     dims_out[RANK];
    hsize_t     num_obj,obj_idx;
    char        dset_name[MAX_NAME];
    size_t      name_size;
    int         dataset_size[10];
    int         dataset_idx[10];
    int         dataset_per_proc;
    double complex     *dset_data;//using malloc is better
    int         otype;
    bool        iscomplex = false;
    int         num_dataset_obj = 0;
    //int         *dataset_obj_idx;
    //fftw_complex *dset_data;


    /* FFTW variables */
    ptrdiff_t n0;
    fftw_plan plan;
    fftw_plan plan2;
    fftw_complex *data;
    fftw_complex *multiplied_fft;
    ptrdiff_t alloc_local, local_n0, local_0_start;//, i, j;
    ptrdiff_t howmany = 2;
    ptrdiff_t local_ni; 
    ptrdiff_t local_i_start; 
    ptrdiff_t local_no; 
    ptrdiff_t local_o_start;

    /*Correlation variables*/
    int num_computed_xcorr = 0;
    int num_xcorr = 0;

    /*
     * MPI variables
     */
    int mpi_size, mpi_rank;
    MPI_Comm comm  = MPI_COMM_WORLD;
    MPI_Info info  = MPI_INFO_NULL;

    /*
     * Initialize MPI
     */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);  

    fftw_mpi_init();
 
    /* 
     * Set up file access property list with parallel I/O access
     */
     plist_id = H5Pcreate(H5P_FILE_ACCESS);
     H5Pset_fapl_mpio(plist_id, comm, info);

    /*
     * Open a new file collectively and release property list identifier.
     */
    file_id = H5Fopen(H5FILE_NAME, H5F_ACC_RDWR, plist_id);
    H5Pclose(plist_id);
    status = H5Gget_num_objs(file_id, &num_obj);
    printf("num_obj is : %d \n", (int) num_obj);
    obj_idx = 0;
    //dataset_id = (hid_t*) malloc(num_obj*sizeof(hid_t))
    while(obj_idx < num_obj)
    {
      otype =  H5Gget_objtype_by_idx(file_id, (size_t)obj_idx);
      if(otype == H5G_DATASET)
      {
          num_dataset_obj = num_dataset_obj +1;
	  H5Gget_objname_by_idx(file_id, obj_idx, dset_name,(size_t)MAX_NAME );
      	  printf("name starts with : %s \n",dset_name);
          printf("obj_idx : %d \n",(int) obj_idx);

      	  /* Open an existing dataset. */
      	  dataset_id = H5Dopen(file_id, dset_name, H5P_DEFAULT);
      
          /* Acquire some dataset information*/
      	  datatype  = H5Dget_type(dataset_id);
          size  = H5Tget_size(datatype);
          printf("size of stored datatype is : %d \n", (int) size);
          printf("size of complex datatype : %lu \n", sizeof(fftw_complex));
          filespace = H5Dget_space(dataset_id);
          rank      = H5Sget_simple_extent_ndims(filespace);
          status  = H5Sget_simple_extent_dims(filespace, dims_out, NULL);
	  printf("dims_out[0] : %d , dims_out[1] : %d \n", (int) dims_out[0], (int) dims_out[1]);
          if(dims_out[0]==1)
		dataset_size[num_dataset_obj-1]=dims_out[1]; 
          else 
		dataset_size[num_dataset_obj-1]=dims_out[0];
       
          dataset_per_proc=(dataset_size[num_dataset_obj-1]/mpi_size)+0.5;   
	  printf("dataset_per_proc : %d \n", (int) dataset_per_proc);
          // Is it first memory space then filespace?
          /*Create the local memory space*/
          
          /*
          count[0]= dataset_per_proc;
          count[1]= 1;
          memspace = H5Screate_simple(RANK,count,NULL);
          dset_data = (double *) malloc(dataset_per_proc*size);
          */
      
          /*Create the filespace/dataset hyperslab*/
          
          if (iscomplex == false){ 		// if dataset is non-complex double values
          	offset[0]=mpi_rank*dataset_per_proc;
          	offset[1]=0;
          	count[0] = dataset_per_proc;
	  	count[1] = 1;
		block[0]= 1; //each block is 1 value representing one non-complex number
                block[1] = 1;
	 }
         else {                                 //if data set is a set of complex numbers where imaginary value is stored right after the real
         	offset[0]=mpi_rank*dataset_per_proc*2;// starting offset is multiplied by two because each complex number is two values
          	offset[1]=0;
          	count[0] = dataset_per_proc; // represents the number of complex numbers
	  	count[1] = 1;
		block[0]= 2; //each block of 2 values represents one complex number
                block[1] = 1;	

	}
          //printf("offset[0] : %llu \n", offset[0]);
          status = H5Sselect_hyperslab(filespace,H5S_SELECT_SET,offset,NULL,count,block);       

          
          /* Create memory space and allocate data buffer*/
	  tot_proc_datasets[0]= dataset_per_proc * howmany * 2; // number of columns, multiplied by two to hold complex
          tot_proc_datasets[1]= 1; // number of rows 
          if(obj_idx == 0){  // create memory space and allocate buffer
               memspace = H5Screate_simple(2,tot_proc_datasets,NULL); //could this go outside loop?
               dset_data = (double complex *)malloc(tot_proc_datasets[0]*size);//always double type data
          }

          /*Create the memory hyperslab to transfer data from the currently opened dataset*/   

          //count[0]= dataset_per_proc; 
          //count[1] = 1;
          offset[0]= (num_dataset_obj-1)*2; //num_dataset_obj-1 is the index of the dataset object if a list containing only dataset objects existed 
          offset[1] = 0;
          stride[0]= num_obj*2;
          stride[1] = 1;
          
          //block[0]= 2; //each block of 2 values represents one complex number
          //block[1] = 1;
          status = H5Sselect_hyperslab(memspace,H5S_SELECT_SET,offset,stride,count,block);


          plist_id = H5Pcreate(H5P_DATASET_XFER);
          H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
      
         // if(obj_idx == 0){
           //  dset_data = (double complex *)malloc(num_obj*dataset_per_proc*size);//always double type data
             //dset_data = (datatype *)malloc(num_obj*size_per_proc*size); 
         // }
      
          /* Read the dataset. */
          status = H5Dread(dataset_id, H5T_IEEE_F64LE, memspace, filespace, H5P_DEFAULT, dset_data);

          if(mpi_rank == 0 && obj_idx == 1){
    	    for (i=0; i<20;i++){
        	    printf("%f",creal(dset_data[1999-i]));
		    printf("\n ");
    	    }
          }

          ++obj_idx;
     
          /* Close/release resources HDF dataset related resources*/
          H5Dclose(dataset_id);
          H5Sclose(filespace);
        }
      
      
    }

    /*HDF5 file read end*/
    
   

   
    /* get local data size and allocate */
    howmany = 2;
    n0 = 2*dataset_size[0];  // here data_size[0] should be replaced by the maximum of datasize array elements
    //tot_buffer_size = howmany*n0;
      //local_ni = proc_size;
    //local_i_start = offset[0];

    printf("fftw n0 is %d \n",(int) n0);
    alloc_local = fftw_mpi_local_size_many_1d(n0, howmany, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE, &local_ni, &local_i_start, &local_n0, &local_0_start);
    printf("local_ni : %d \n",(int) local_ni);
    printf("local_i_start : %d \n",(int) local_i_start);
    printf("local_n0 : %d \n",(int) local_n0);
    printf("local_0_start : %d \n",(int) local_0_start);

    data = fftw_alloc_complex(alloc_local);

    printf("fftw allocated size is %d \n",(int) alloc_local);

    

   
    //for one-dimensional transforms, only composite (non-prime) n0 are currently supported (unlike the serial FFTW). Requesting an unsupported transform size will yield a NULL plan. (As in the serial interface, highly composite sizes generally yield the best performance.)  
    /* create plan for in-place forward DFT */
    plan = fftw_mpi_plan_many_dft(1, &n0, howmany,FFTW_MPI_DEFAULT_BLOCK,FFTW_MPI_DEFAULT_BLOCK, data, data, MPI_COMM_WORLD,FFTW_FORWARD, FFTW_ESTIMATE); 
   if(plan == NULL)
      printf("plan is null \n");

    /* initialize data to some function my_function(x,y) */
    printf("Input  data: \n");
    //data = (fftw_complex *) dset_data;
  
    if(local_i_start + local_ni-1 < dataset_size[0]) //not yet reached 0 appendinging region
    {
    	for (i = 0; i < howmany*local_ni; i++) {
            	data[i] = dset_data[i]; 
            	if(i<10) printf("  %f + i%f \n", creal(data[i]), cimag(data[i]));
     	}
    }

    if(local_i_start < dataset_size[0] && local_i_start + local_ni-1 > dataset_size[0] )  //dataset size 0 used but normally should be the maximum value in the datasert size array
    {
        printf("  Inside second if part \n");
    	for (i = 0; i < howmany * dataset_size[0]; i++) {
            	data[i] = dset_data[i]; 
            	//if(i<10) printf("  %f + i%f \n", creal(data[i]), cimag(data[i]));
     	}
	
	for (i = howmany*dataset_size[0]; i < howmany*local_ni; i++) { //append zeros for the final data values
            	data[i] =  0; 
            	//if(i<10) printf("  %f + i%f \n", creal(data[i]), cimag(data[i]));
     	}
    }

    if(local_i_start > dataset_size[0]) //append zeros for the final data values
    {
    	for (i = 0; i < howmany*local_ni; i++) {
            	data[i] = 0; 
            	if(i<10) printf("  %f + i%f \n", creal(data[i]), cimag(data[i]));
     	}
    }

  /* if(i = 0; i < howmany*local_ni; i++) //not yet reached 0 appendinging region
    {
    	for (i = 0; i < howmany*local_ni; i++) {
            	data[i] = dset_data[i]; 
            	if(i<10) printf("  %f + i%f \n", creal(data[i]), cimag(data[i]));
     	}
    }*/

   /* compute transforms, in-place, as many times as desired */
   fftw_execute(plan);

   printf("output  data: \n");
   if(mpi_rank == 0){
    	for (i=0; i<10;i++){ //originally checking for 3990 - 4000 
        	printf("  %f + i%f \n", creal(data[i]), cimag(data[i]));
    	}
    }
 
   fftw_destroy_plan(plan); 
   
  
   /* DFT multiply section begins*/
   for(i=0; i<howmany; i++) {
  	num_xcorr=num_xcorr+i;
        printf("  number of correlations: %d \n", num_xcorr);
   }
    
     
   
   int length_fft = n0;
   int signal_num = howmany;
   howmany = num_xcorr;
   
   n0 = n0*num_xcorr;

    printf("n0 is: %d \n",(int)n0);
   /*Determine required space and allocate memory space for fft computation*/
   alloc_local = fftw_mpi_local_size_many_1d(n0, howmany, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE, &local_ni, &local_i_start, &local_n0, &local_0_start);

   multiplied_fft = fftw_alloc_complex(alloc_local);
   printf("alloc local is: %d \n",(int) alloc_local);

   /* create plan for in-place forward DFT */
    plan2 = fftw_mpi_plan_many_dft(1, &n0, howmany,FFTW_MPI_DEFAULT_BLOCK,FFTW_MPI_DEFAULT_BLOCK, multiplied_fft, multiplied_fft, MPI_COMM_WORLD,FFTW_BACKWARD, FFTW_ESTIMATE); 

   int j;
   int set_base_idx = 0;
   int index = 0;
   while(set_base_idx < signal_num*length_fft) //set index - begining of a signal fft group with the same index before multiplication
   {
	for(i=set_base_idx;i<set_base_idx+signal_num-1;i++)
	{
	    for(j=i+1;j<set_base_idx+signal_num; j++)
	       multiplied_fft[index]=data[i]*data[j];
               index = index+1;
	}
	set_base_idx = set_base_idx+signal_num;
        //base = base + num_xcorr;
        
        //if(base%100 == 0)  printf("  base is: %d \n", base);;
   }
   
   
   /* DFT multiply section ends*/

   printf("index is: %d \n",index);
   printf("set base index is: %d \n",set_base_idx);
   printf("output after fft multiplication: \n");
   if(mpi_rank == 0){
    	for (i=0; i<10;i++){
        	printf("  %f + i%f \n", creal(multiplied_fft[i]), cimag(multiplied_fft[i]));
    	}
    }
  

   fftw_execute(plan2);

   printf("output after inverse fft: \n");
   if(mpi_rank == 0){
    	for (i=0; i<35;i++){ //originally checking for 3990 - 4000 
        	printf("  %f + i%f \n", creal(multiplied_fft[i]), cimag(multiplied_fft[i]));
    	}
    }

    fftw_destroy_plan(plan2);
    fftw_mpi_cleanup();
    free(dset_data);

    /*
     * Close/release resources.
     */
    H5Pclose(plist_id);
    H5Fclose(file_id);
 
    MPI_Finalize();

    return 0;
}     
