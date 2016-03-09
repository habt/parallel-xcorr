
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
#define MAX_NUM_DATASETS 10
#define MAX_NUM_XCORR 45        // with 10 signals this value is equal to 9+8+7+6+5+4+3+2+1 

int
main (int argc, char **argv)
{
    /*
     * HDF5 APIs definitions
     */ 	
    hid_t       file_id; 
    hid_t       dataset_id[MAX_NUM_DATASETS];               /* file and dataset identifiers */
    hid_t       filespace[MAX_NUM_DATASETS];
    hid_t       memspace;      /* file and memory dataspace identifiers */
    hsize_t     dimsf[RANK];                 /* dataset dimensions */ 
    hsize_t     tot_proc_datasets[RANK];	         
    hsize_t	count[RANK];                 /* hyperslab selection parameters */
    hsize_t	offset[RANK];
    hsize_t     stride[RANK];
    hsize_t     block[RANK];
    hid_t	plist_id;                 /* property list identifier */
    int         i,j;
    herr_t	status;
    hssize_t 	data_dim;
    size_t      size; 
    hid_t       datatype;
    int         rank;
    hsize_t     dims_out[RANK];
    hsize_t     num_obj;
    hsize_t     obj_idx;
    char        dset_name[MAX_NAME];
    size_t      name_size;
    int         dataset_size[MAX_NUM_DATASETS];
    int         xcorr_size[MAX_NUM_XCORR];
    hsize_t     dataset_idx[MAX_NUM_DATASETS];
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

    /* Other Correlation variables*/
    int num_computed_xcorr = 0;
    int num_xcorr = 0;
    int set = 1;
    int max_size = 0;
    int sec_max_size = 0;
    int length_fft;
    int signal_num;
    bool is_data_read = true;

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




/*******************************************************************************************/
    //dataset_id = (hid_t*) malloc(num_obj*sizeof(hid_t))
    while(obj_idx < num_obj)
    {
      otype =  H5Gget_objtype_by_idx(file_id, (size_t)obj_idx);
      if(otype == H5G_DATASET)
      {
          dataset_idx[num_dataset_obj] = obj_idx;
          num_dataset_obj = num_dataset_obj +1;
      }
      obj_idx++;
    }



/******************************************************************************************/
    /* Loop to determine the biggest dataset size to determine the number of zeros to be appended to each signal before the FFT*/
    /* an alternative is set a maximum length for correlation and extend each signal to tat length */
    for(i=0;i<num_dataset_obj; i++) // NB this loop can be removed and the code written inside the while loop above
    {
        printf("loop : %d \n", i);
	H5Gget_objname_by_idx(file_id, dataset_idx[i], dset_name,(size_t)MAX_NAME );
        dataset_id[i] = H5Dopen(file_id, dset_name, H5P_DEFAULT);
        printf("data set name %d is %s  \n", i,dset_name);

	filespace[i] = H5Dget_space(dataset_id[i]);
        rank      = H5Sget_simple_extent_ndims(filespace[i]);
        status  = H5Sget_simple_extent_dims(filespace[i], dims_out, NULL);
	printf("dims_out[0] : %d , dims_out[1] : %d \n", (int) dims_out[0], (int) dims_out[1]);
        if(dims_out[0]==1)
           dataset_size[i]=dims_out[1]; 
        else 
	   dataset_size[i]=dims_out[0];
        
        if(dataset_size[i]>sec_max_size)
	{
            if(dataset_size[i]>max_size)
     	    {
		max_size=dataset_size[i];
	        sec_max_size = max_size;
	    }
	    else
	       sec_max_size = dataset_size[i];
	}
	
    }
    
    /* get local data size and allocate */
    howmany =num_dataset_obj;
    printf("MPI size : %d \n", mpi_size);
    n0 = max_size+sec_max_size;  // here data_size[0] should be replaced by the maximum of datasize array elements
    dataset_per_proc=(n0/mpi_size)+0.5;   
    printf("dataset_per_proc : %d \n", (int) dataset_per_proc);
    printf("howmany is : %d \n", (int) howmany);

    /* Create memory space and allocate data buffer*/
    tot_proc_datasets[0]= dataset_per_proc * howmany * 2; // number of columns, multiplied by two to hold complex
    tot_proc_datasets[1]= 1; // number of rows 
    memspace = H5Screate_simple(2,tot_proc_datasets,NULL);
   // dset_data = (double complex *)malloc(tot_proc_datasets[0]*size);//always double type data

    
    printf("fftw n0 is %d \n",(int) n0);
    alloc_local = fftw_mpi_local_size_many_1d(n0, howmany, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE, &local_ni, &local_i_start, &local_n0, &local_0_start);

    printf("local_ni : %d \n",(int) local_ni);
    printf("local_i_start : %d \n",(int) local_i_start);
    printf("local_n0 : %d \n",(int) local_n0);
    printf("local_0_start : %d \n",(int) local_0_start);

    printf("fftw allocated size is %d \n",(int) alloc_local);

    data = fftw_alloc_complex(alloc_local);

    //printf("fftw allocated size is %d \n",(int) alloc_local);
    
    /* create plan for in-place forward DFT */
    plan = fftw_mpi_plan_many_dft(1, &n0, howmany,FFTW_MPI_DEFAULT_BLOCK,FFTW_MPI_DEFAULT_BLOCK, data, data, MPI_COMM_WORLD,FFTW_FORWARD, FFTW_ESTIMATE); 
   if(plan == NULL) printf("plan is null \n");
    
    if(iscomplex == true) set = 2;

     


    /************************************************************************************/
    for(i=0;i<num_dataset_obj; i++)
    { 
	   //if(i=0) data = fftw_alloc_complex(alloc_local);
 
          /*Create the filespace/dataset hyperslab*/
          printf("dataset[%d] size is %d \n",i,(int) dataset_size[i]);
          if(local_i_start + local_ni <= dataset_size[i])
	  {                               
         	offset[0]=mpi_rank*local_ni*set;// starting offset is multiplied by two because each complex number is two values
          	offset[1]=0;
          	count[0] = local_ni; // represents the number of complex numbers
	  	count[1] = 1;
		block[0]= set; //each block of 2 values represents one complex number
                block[1] = 1;
	        status = H5Sselect_hyperslab(filespace[i],H5S_SELECT_SET,offset,NULL,count,block);
                is_data_read = true;	
	  }
	  
	  if(local_i_start < dataset_size[i] && local_i_start + local_ni > dataset_size[i])
	  {                               
         	offset[0]=mpi_rank*local_ni*set;// starting offset is multiplied by two because each complex number is two values
          	offset[1]=0;
          	count[0] = dataset_size[i]-local_i_start ; // represents the number of complex numbers
	  	count[1] = 1;
		block[0]= set; //each block of 2 values represents one complex number
                block[1] = 1;
                status = H5Sselect_hyperslab(filespace[i],H5S_SELECT_SET,offset,NULL,count,block);
                is_data_read = true;	
	  }
		
	  if(local_i_start >= dataset_size[i]) //all zeros
	  {
	       for(j=0;j<local_ni; j++)
    	       {
		  data[i]= 0.0;
	       }
               
               is_data_read = false;

	  }
          
          printf("process %d, dataset %d,count[0] : %llu \n", mpi_rank, i, count[0]);
          
          /*Create the memory hyperslab to transfer data from the currently opened dataset*/   

          if (is_data_read)
	  {
	     offset[0]= i*2; //i is the index of the dataset object if a list containing only dataset objects existed 
             offset[1] = 0;
             stride[0]= num_dataset_obj*2;
             stride[1] = 1;
             block[0]= 1; //each block of 2 values represents one complex number
             block[1] = 1;
          
             status = H5Sselect_hyperslab(memspace,H5S_SELECT_SET,offset,stride,count,block);

             
             plist_id = H5Pcreate(H5P_DATASET_XFER);
             H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
      
      
             /* Read the dataset. */
             status = H5Dread(dataset_id[i], H5T_IEEE_F64LE, memspace, filespace[i], H5P_DEFAULT, data);
     	  }
          

          if(mpi_rank == 0 && i == 1){
    	    for (j=0; j<10;j++){
        	    printf("%f",creal(data[j]));
		    printf("\n ");
    	    }
          }
     
          /* Close/release resources HDF dataset related resources*/
          H5Dclose(dataset_id[i]);
          H5Sclose(filespace[i]);
      
      
    }

    /*HDF5 file read end*/
    printf("process %d Input  data: \n",mpi_rank);
    //data =   dset_data;
    for (i = 0; i < 10; i++) {
         if(i<10) printf("  %f + i%f \n", creal(data[i]), cimag(data[i]));
     }

   /* compute transforms, in-place, as many times as desired */
   fftw_execute(plan);

   printf("process %d FFT output  data: \n",mpi_rank);
   if(mpi_rank == 1){
    	for (i=0; i<10;i++){ //originally checking for 3990 - 4000 
        	printf("  %f + i%f \n", creal(data[i]), cimag(data[i]));
    	}
    }
 
   fftw_destroy_plan(plan); 
   
  /*************************************************************************************/
   /* DFT multiply section begins*/
   for(i=0; i<howmany; i++) {
  	num_xcorr=num_xcorr+i;
   }
    
   printf(" number of correlations: %d \n", num_xcorr); 
   
   length_fft = local_n0;
   signal_num = howmany;
   howmany = num_xcorr;
   
   n0 = n0*num_xcorr;

    printf("process %d n0 is: %d \n",mpi_rank, (int)n0);
   /*Determine required space and allocate memory space for fft computation*/
   alloc_local = fftw_mpi_local_size_many_1d(n0, howmany, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE, &local_ni, &local_i_start, &local_n0, &local_0_start);

   multiplied_fft = fftw_alloc_complex(alloc_local);
   printf("alloc local is: %d \n",(int) alloc_local);

   /* create plan for in-place forward DFT */
    plan2 = fftw_mpi_plan_many_dft(1, &n0, howmany,FFTW_MPI_DEFAULT_BLOCK,FFTW_MPI_DEFAULT_BLOCK, multiplied_fft, multiplied_fft, MPI_COMM_WORLD,FFTW_BACKWARD, FFTW_ESTIMATE); 




   /**************************************************************************************************/
   int set_base_idx = 0;
   int index = 0;
   while(set_base_idx < signal_num*length_fft) //set index - begining of a signal fft group with the same index before multiplication
   {
	for(i=set_base_idx;i<set_base_idx+signal_num-1;i++)
	{
	    for(j=i+1;j<set_base_idx+signal_num; j++)
	       multiplied_fft[index]=data[i]*conj(data[j]);
               index = index+1;
	}
	set_base_idx = set_base_idx+signal_num;
        //base = base + num_xcorr;
        
        //if(base%100 == 0)  printf("  base is: %d \n", base);;
   }
   
   
   /* DFT multiply section ends*/

   printf("process %d index is: %d \n",mpi_rank, index);
   printf("process %d set base index is: %d \n",mpi_rank,set_base_idx);
   printf("process %d output after fft multiplication: \n", mpi_rank);
   if(mpi_rank == 1){
    	for (i=0; i<10;i++){
        	printf("  %f + i%f \n", creal(multiplied_fft[i]), cimag(multiplied_fft[i]));
    	}
    }
  

   fftw_execute(plan2);

   printf("output after inverse fft: \n");
   if(mpi_rank == 1){
    	for (i=0; i<10;i++){ //originally checking for 3990 - 4000 
        	printf("  %f + i%f \n", creal(multiplied_fft[i]), cimag(multiplied_fft[i]));
    	}
    }

    fftw_destroy_plan(plan2);
    fftw_mpi_cleanup();
    free(dset_data);

    /*
     * Close/release resources.
     */
    if (is_data_read) {H5Pclose(plist_id);}
    
    H5Fclose(file_id);
 
    MPI_Finalize();

    return 0;
}     
