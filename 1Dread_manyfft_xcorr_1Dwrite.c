#include <unistd.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "stdlib.h"
#include "stdbool.h"
#include <complex.h> 
#include "hdf5.h"
#include <mpi.h>
#include <fftw3-mpi.h>
#include <memory.h>
#include <time.h>


#define H5FILE_NAME     "testfile.h5"
#define DATASETNAME 	"X_Dataset" 
#define NX     1                     /* dataset dimensions */
#define NY     1000
#define RANK   2
#define MAX_NAME 20
#define MAX_NUM_DATASETS 10
#define MAX_NUM_XCORR 45        // with 10 signals this value is equal to 9+8+7+6+5+4+3+2+1 

int
main (int argc, char* argv[])
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
    int         actual_data_size[MAX_NUM_DATASETS];
    int         xcorr_size[MAX_NUM_XCORR];
    hsize_t     dataset_idx[MAX_NUM_DATASETS];
    int         dataset_per_proc;
    double complex     *dset_data;//using malloc is better
    int         otype;
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
    int iscomplex = 0;
    int read_set = 1;
    int write_set = 2;
    int max_size = 0;
    int sec_max_size = 0;
    int length_fft;
    int signal_num;
    bool is_data_read = true;
    unsigned int compute_flag;
    
    

    /* Time Variables */
    clock_t start, end;
    double elapsed_times[7];

    /*
     * MPI variables
     */
    int mpi_size, mpi_rank;
    MPI_Comm comm  = MPI_COMM_WORLD;
    MPI_Info info  = MPI_INFO_NULL;

    /*File variables*/
    FILE *outfptr;
    char       outf_name[100];
    char       inph5_name[100] = "testfile.h5";
    char       perf_out[1000];
    char       interm_str[100];
    int        length_count = 0;
    char str[] = " MPIsize  ReadHDF(*datasetnumber)   PlanFWDFFTW   ExecuteFFTW   Multiplication  PlanBWDFFTW  ExecuteFFTW  Filename";
    
    
    
    /* Start by interpreting command line arguements*/
    
   printf("Number of arguements is %d \n",argc);
   
   
    sprintf(inph5_name,"%s%s", argv[5], argv[4]);
    printf("Input file name is %s \n",inph5_name);

      
    //strncpy(perf_out,argv[4],strlen(argv[4])-3);
    //memcpy(perf_out,argv[4],sizeof(argv[4])-3);
    

    if(strcmp(argv[1],"fl")==0 ) //testing for FFTW flag performances
    {
        if(strcmp(argv[2],"es")==0) compute_flag = FFTW_ESTIMATE; 
        if(strcmp(argv[2],"me")==0) compute_flag = FFTW_MEASURE;
        if(strcmp(argv[2],"pa")==0) compute_flag = FFTW_PATIENT;
        if(strcmp(argv[2],"ex")==0) compute_flag = FFTW_EXHAUSTIVE;
        printf("argument 1 is  ' %s '  \n", argv[1]);       
       
    }
    
    else if(strcmp(argv[1],"nf")==0) //testing for non-flag performances
    {
	compute_flag = FFTW_ESTIMATE;
        printf("flag is nf  \n");
    }
    
    else printf("Improper arguement 1: NB: choose input either 'fl' or 'nf' \n");
 

    if(strcmp(argv[3],"clx")==0) 
    {
	iscomplex = 1;
   	printf("Input is complex, set iscomplex to %d  \n",iscomplex);
    }
       
    int num =0;
    for(num = 0; num < argc; num++)
        printf("argv[%d] = %s \n",num, argv[num]);
    
    printf("output text file name is '%s'  \n", outf_name); 
    
    
    /*
     * 0. executable 1. flag test or not  
     * 2. which flag or type of test 3. double or complex 
     * 4.input hdf file 5.path to hdf file and output text file
    */
    if(argc != 6) 
      { 
	printf("Improper number of arguements \n"); 
        exit(1);
      }
    /*
     * Initialize MPI
     */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(comm, &mpi_size);
    MPI_Comm_rank(comm, &mpi_rank);  

    fftw_mpi_init();
 
    sprintf(outf_name,"%soutputs/%s_%s_%s_%d.txt",argv[5], argv[1],argv[2],argv[3],mpi_rank); //fopen() requires full path name starting from root location-unlike HDF open paths

       
    //sprintf(interm_str,"%d      ",mpi_size);
    sprintf(perf_out,"  %d  ",mpi_size);
    //strncat(perf_out,interm_str,strlen(interm_str));
    //length_count = length_count + strlen(perf_out);
    /* 
     * Set up file access property list with parallel I/O access
     */
     plist_id = H5Pcreate(H5P_FILE_ACCESS);
     H5Pset_fapl_mpio(plist_id, comm, info);

    /*
     * Open an HDF5 file collectively and release property list identifier.
     */
    file_id = H5Fopen(inph5_name, H5F_ACC_RDWR, plist_id);
    H5Pclose(plist_id);
    status = H5Gget_num_objs(file_id, &num_obj);
    printf("num_obj is : %d \n", (int) num_obj);
    obj_idx = 0;

    /* 
     *  Iterate through the file and determine the number of dataset objects 
     */
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

        actual_data_size[i] = dataset_size[i];
/* NB this is the count of double data types/ if the file is supposed to store complex the actual number of values is half of this*/
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
    if(iscomplex == 0) n0 = max_size+sec_max_size;  // divide this by two if complex dataset
    else n0 = (max_size+sec_max_size)*0.5;

    /* Create memory space and allocate data buffer*/
    alloc_local = fftw_mpi_local_size_many_1d(n0, howmany, MPI_COMM_WORLD, FFTW_FORWARD, compute_flag, &local_ni, &local_i_start, &local_n0, &local_0_start);
    printf("max_size is : %d \n", max_size);
    printf("MPI size : %d \n", mpi_size);
    printf("howmany is : %d \n", (int) howmany);
    printf("fftw n0 is %d \n",(int) n0);
    printf("Process %d local_ni : %d \n", mpi_rank, (int) local_ni);
    printf("Process %d local_i_start : %d \n", mpi_rank,(int) local_i_start);
    printf("Process %d local_n0 : %d \n",mpi_rank,(int) local_n0);
    printf("Process %d local_0_start : %d \n", mpi_rank,(int) local_0_start);

    printf("fftw allocated size is %d \n",(int) alloc_local);

    data = fftw_alloc_complex(alloc_local);
  
    /* create plan for in-place forward DFT */
    start = clock();
    plan = fftw_mpi_plan_many_dft(1, &n0, howmany,FFTW_MPI_DEFAULT_BLOCK,FFTW_MPI_DEFAULT_BLOCK, data, data, MPI_COMM_WORLD,FFTW_FORWARD, compute_flag); 
    end = clock();
    int x = 0;
    elapsed_times[x] = (end-start)/CLOCKS_PER_SEC;
    x= x+1;    
    if(plan == NULL) printf("plan is null \n");
    
    if(iscomplex == 1) {read_set = 2;}

    tot_proc_datasets[0]= local_ni * howmany * 2; // number of columns, multiplied by two to hold complex
    tot_proc_datasets[1]= 1; // number of rows 
    memspace = H5Screate_simple(2,tot_proc_datasets,NULL);

     printf("read set is %d \n",read_set);

    if(iscomplex == 1)
    {
    	 for(i=0;i<num_dataset_obj; i++)
     	 {
		actual_data_size[i] = dataset_size[i]*0.5;
	 }
    }

    /************************************************************************************/
    for(i=0;i<num_dataset_obj; i++)
    { 
          /*Create the filespace/dataset hyperslab*/
          printf("before filespace hyperslab dataset[%d] size is %d \n",i,(int) dataset_size[i]);
          if(local_i_start + local_ni <= actual_data_size[i]) // tis dataset size used here has to be halved for complex
	  {                               
         	offset[0]=local_i_start*read_set;// starting offset is multiplied by two because each complex number is two values
          	offset[1]=0;
          	count[0] = local_ni*read_set; // represents the number of complex numbers
	  	count[1] = 1;
		block[0]= 1; //each block of 2 values represents one complex number
                block[1] = 1;
	        status = H5Sselect_hyperslab(filespace[i],H5S_SELECT_SET,offset,NULL,count,block);
                is_data_read = true;	
	  }
	  
	  if(local_i_start < actual_data_size[i] && local_i_start + local_ni > actual_data_size[i])
	  {                               
         	offset[0]=local_i_start*read_set;// starting offset is multiplied by two because each complex number is two values
          	offset[1]=0;
          	count[0] = (actual_data_size[i]-local_i_start)*read_set ; // represents the number of complex numbers
	  	count[1] = 1;
		block[0]= 1; //each block of 2 values represents one complex number
                block[1] = 1;
                status = H5Sselect_hyperslab(filespace[i],H5S_SELECT_SET,offset,NULL,count,block);
                is_data_read = true;	
	  }
		
	  if(local_i_start >= actual_data_size[i]) //all zeros
	  {
               start = clock();
              
	       for(j=0;j<local_ni; j++)
    	       {
		  data[i]= 0.0;
	       }
               end = clock();
               elapsed_times[x] = (end-start)/CLOCKS_PER_SEC;
               x= x+1;
               //sprintf(interm_str," %f ",elapsed_time);
               //sprintf(perf_out+sizeof(perf_out),"%f",elapsed_time);
               //strncat(perf_out,interm_str,strlen(interm_str));
               //printf("Process %d : Data read time is  %f seconds \n", mpi_rank, elapsed_time);
               is_data_read = false;

	  }
          
          printf("process %d, dataset %d,count[0] : %llu \n", mpi_rank, i, count[0]);
          /*Create the memory hyperslab to transfer data from the currently opened dataset*/   
          printf("before memoryspace hyperslab dataset[%d] size is %d \n",i,(int) dataset_size[i]);
          if (is_data_read)
	  {
	     offset[0]= i*write_set; //i is the index of the dataset object if a list containing only dataset objects existed 
             offset[1] = 0;
             stride[0]= num_dataset_obj*write_set;
             stride[1] = 1;
             if(iscomplex == 1){block[0]= 2;count[0] = count[0]*0.5;}
             status = H5Sselect_hyperslab(memspace,H5S_SELECT_SET,offset,stride,count,block);

	     printf("before READ dataset[%d] size is %d \n",i,(int) dataset_size[i]);      
             /* Read the dataset. */
             start = clock();
             status = H5Dread(dataset_id[i], H5T_IEEE_F64LE, memspace, filespace[i], H5P_DEFAULT, data);
             end = clock();
             elapsed_times[x] = (end-start)/CLOCKS_PER_SEC;
             x= x+1;
             //sprintf(interm_str," %f ",elapsed_time);
             //sprintf(perf_out+sizeof(perf_out),"%f",elapsed_time);
             //strncat(perf_out,interm_str,strlen(interm_str));
             //printf("Process %d : Data read time is  %f seconds \n", mpi_rank, elapsed_time); 
    	  }
     
          /* Close/release resources HDF dataset related resources*/
          H5Dclose(dataset_id[i]);
          H5Sclose(filespace[i]);
      
      
    }
    H5Sclose(memspace);

    /*HDF5 file read end*/
    printf("process %d Input  data: \n",mpi_rank);
    
    for (i = 0; i < 10; i++) {
         if(i<10) printf("  %f + i%f \n", creal(data[i]), cimag(data[i]));
     }

   /* compute transforms, in-place, as many times as desired */
   start = clock();
   fftw_execute(plan);
   end = clock();
   elapsed_times[x] = (end-start)/CLOCKS_PER_SEC;
    x= x+1;
   //sprintf(interm_str," %f ",elapsed_time);
   //sprintf(perf_out+sizeof(perf_out),"%f",elapsed_time);
   //sprintf(perf_out+sizeof(perf_out),"%s",interm_str);
   //strncat(perf_out,interm_str,strlen(interm_str));
   printf("Process %d : forward FFTW execution time is %f seconds \n", mpi_rank, elapsed_times[x-1]);
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
   

    printf("process %d n0 is: %d \n",mpi_rank, (int)n0);
   /*Determine required space and allocate memory space for fft computation*/
   alloc_local = fftw_mpi_local_size_many_1d(n0, howmany, MPI_COMM_WORLD, FFTW_BACKWARD, compute_flag, &local_ni, &local_i_start, &local_n0, &local_0_start);

   multiplied_fft = fftw_alloc_complex(alloc_local);
   printf("alloc local is: %d \n",(int) alloc_local);
    printf("inverse FFT process %d local n0 is: %d \n",mpi_rank, (int)local_n0);
    printf("inverse FFT process %d local ni is: %d \n",mpi_rank, (int)local_ni);

   /* create plan for in-place forward DFT */
   start = clock();
    plan2 = fftw_mpi_plan_many_dft(1, &n0, howmany,FFTW_MPI_DEFAULT_BLOCK,FFTW_MPI_DEFAULT_BLOCK, multiplied_fft, multiplied_fft, MPI_COMM_WORLD,FFTW_BACKWARD, compute_flag); 
   
   end = clock();
   elapsed_times[x] = (end-start)/CLOCKS_PER_SEC;
    x= x+1;
   //printf("Process %d : backward FFTW planning time is %f seconds \n", mpi_rank, elapsed_time);
   //sprintf(perf_out+sizeof(perf_out),"%f",elapsed_time);
   //sprintf(interm_str," %f ",elapsed_time);
   //sprintf(perf_out+sizeof(perf_out),"%s",interm_str);
   //strncat(perf_out,interm_str,strlen(interm_str));




   /**************************************************************************************************/
   int set_base_idx = 0;
   int index = 0;
   end = clock();
   while(set_base_idx < signal_num*length_fft) //set index - begining of a signal fft group with the same index before multiplication
   {
	for(i=set_base_idx;i<set_base_idx+signal_num-1;i++)
	{
	    for(j=i+1;j<set_base_idx+signal_num; j++)
	       multiplied_fft[index]=data[i]*conj(data[j]);
               index = index+1;
	}
	set_base_idx = set_base_idx+signal_num;
   }
 
   end = clock();
   elapsed_times[x] = (end-start)/CLOCKS_PER_SEC;
    x= x+1;
   //printf("Process %d : Multiplication time is %f seconds \n", mpi_rank, elapsed_time);
   //sprintf(perf_out+sizeof(perf_out),"%f",elapsed_time);
   //sprintf(interm_str," %f ",elapsed_time);
   //sprintf(perf_out+sizeof(perf_out),"%s",interm_str);
   //strncat(perf_out,interm_str,strlen(interm_str));
   
   /* DFT multiply section ends*/

   printf("process %d index is: %d \n",mpi_rank, index);
   printf("process %d set base index is: %d \n",mpi_rank,set_base_idx);
   printf("process %d output after fft multiplication: \n", mpi_rank);
   if(mpi_rank == 1){
    	for (i=0; i<10;i++){
        	printf("  %f + i%f \n", creal(multiplied_fft[i]), cimag(multiplied_fft[i]));
    	}
    }
     
   start = clock();
   fftw_execute(plan2);
   end = clock();
   elapsed_times[x] = (end-start)/CLOCKS_PER_SEC;
    x= x+1;
   //printf("Process %d : backward FFTW execusion time is %f seconds \n", mpi_rank, elapsed_time);
   //sprintf(perf_out+sizeof(perf_out),"%f",elapsed_time);
   //sprintf(interm_str," %f",elapsed_time);
   //sprintf(perf_out+sizeof(perf_out),"%s",interm_str);
   //strncat(perf_out,interm_str,strlen(interm_str));
   //sprintf(outf_name,"%s%s_%s_%s.txt",argv[5], argv[1],argv[2],argv[3]); //fopen() requires full path name starting from root location-unlike HDF open paths
int b;
   for (b=0; b<x;b++){
      sprintf(interm_str,"%f  ",elapsed_times[b]);
      printf(" concatenated is %s \n",perf_out);
      printf(" to be concatenated is %f with length %d from %d values\n",elapsed_times[b],(int)strlen(interm_str),x);//(int)strlen(interm_str)
      //sprintf(perf_out+12,",%f",elapsed_times[b]);
      //sprintf(perf_out+sizeof(perf_out),"%s",interm_str);
       strncat(perf_out,interm_str,strlen(interm_str));
   }
   strncat(perf_out,argv[4],strlen(argv[4]));
    //length_count = length_count + strlen(argv[4]);
   printf(" length count is %d \n",length_count);
   //length_count = length_count + b*strlen(interm_str);
   length_count = length_count + strlen(perf_out);
   printf(" amount to be written to file is %d \n",length_count);
   printf(" file is %s \n",outf_name);
   printf("Process %d : printing time parameters: %s, to file %s \n",mpi_rank, perf_out,outf_name);
   

   if (file_exist (outf_name))
   {
      printf ("file exists\n");
      outfptr=(fopen(outf_name,"a"));
   }
   else 
   {
     outfptr=(fopen(outf_name,"w")); // \r\n
     fwrite(str, 1 , strlen(str) , outfptr);
     fwrite("\n", sizeof(char), 1, outfptr);
   }
   if(outfptr==NULL){
       printf("Error opening file!");
       exit(1);
   }
   fwrite(perf_out, 1 , length_count , outfptr);//sizeof(perf_out)*5-6 = 44 is total byte size of the values and the commas where sizeof(perf_out) = 10
   //fwrite(" \r\n", 1 , sizeof(" \r\n"), outfptr);
   fprintf(outfptr, "\n"); 
   fclose(outfptr);
   

   printf("output after inverse fft: \n");
   if(mpi_rank == 0){
    	for (i=0; i<10;i++){ //originally checking for 3990 - 4000 
        	printf("  %f + i%f \n", creal(multiplied_fft[i]), cimag(multiplied_fft[i]));
    	}
    }


    //writetohdf(multiplied_fft,argv[4],argv[5],num_xcorr,local_n0,comm,mpi_rank, mpi_size);
    
    printf("after inverse fft print: \n");
    fftw_destroy_plan(plan2);
    fftw_mpi_cleanup();
    free(dset_data);

    /*
     * Close/release resources.
     */
    //if (is_data_read) {H5Pclose(plist_id);}
    
   H5Fclose(file_id);
 
    MPI_Finalize();

    return 0;
}    

int file_exist (char *filename)
{
  struct stat   st;   
  return (stat (filename, &st) == 0);
}

int  writetohdf(double complex* multiplied_fft,char* inpfilename,char* filepath, int num_signals, int signal_length, MPI_Comm comm, int mpi_rank, int mpi_size)
{
      hid_t	plist2_id; 
      hid_t	file2_id;
      hid_t	dataset_id;
      hid_t     dataspace_id;
      hid_t     memspace_id;
      char     outfilename[100]; 
      char     dataset_name[20] = "xcorr";
      hsize_t     dimsd[RANK];
      int i;
      int total_signal_length;
      hsize_t	  count[RANK];                 /* hyperslab selection parameters */
      hsize_t	  offset[RANK];
      hsize_t     stride[RANK];
      hsize_t     block[RANK];
      herr_t	status;
      clock_t start, end;
      double elapsed_time;

      //memcpy(outfilename,inpfilename, strlen(inpfilename)-4);
      sprintf(outfilename,"%s", filepath);
      strncat(outfilename,"outputs/",strlen("outputs/"));
      strncat(outfilename,inpfilename,strlen(inpfilename)-3);
      printf("output file name %s \n",outfilename);
      strcat(outfilename,"_output.h5");
      printf("output file name %s \n",outfilename);

      plist2_id = H5Pcreate(H5P_FILE_ACCESS);
      H5Pset_fapl_mpio(plist2_id, comm, MPI_INFO_NULL);
      file2_id = H5Fcreate(outfilename, H5F_ACC_TRUNC,H5P_DEFAULT, plist2_id);
      H5Pclose(plist2_id);
      MPI_Allreduce(&signal_length, &total_signal_length, 1, MPI_INT, MPI_SUM, comm);
      dimsd[0]= total_signal_length*2;
      dimsd[1]= 1;
      dataspace_id = H5Screate_simple(2,dimsd,NULL); //here there is an error. solution is to add up the signal lengths of different processes. may be use mpi reduce
      dimsd[0]= signal_length*2*num_signals;
      memspace_id = H5Screate_simple(2,dimsd,NULL);
      plist2_id = H5Pcreate(H5P_DATASET_XFER);
      H5Pset_dxpl_mpio(plist2_id, H5FD_MPIO_COLLECTIVE); 
      for(i=0;i<num_signals;i++)
      {
             //printf("Inside for loop  \n");
             sprintf(dataset_name,"_%d",i); 
             printf("Inside for loop  \n");
             
             /*Create dataset and specify filespace/datset space hyperslab parameters*/
             dataset_id = H5Dcreate(file2_id,dataset_name,H5T_IEEE_F64LE,dataspace_id,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);

             offset[0]= mpi_rank*signal_length*2; //i is the index of the dataset object if a list containing only dataset objects existed 
             offset[1] = 0;
             stride[0]= 1;
             stride[1] = 1;
             block[0]= 1; //each block of 2 values represents one complex number
             block[1] = 1;
             count[0]= signal_length*2;
             count[1] = 1;
            
            printf("Before filespace hyperslab selection  \n");
             status = H5Sselect_hyperslab(dataspace_id,H5S_SELECT_SET,offset,NULL,count,block);
             
             /* Define memory space and memory hyperslab selection mechanism*/
             
             offset[0]= i; //i is the index of the dataset object if a list containing only dataset objects existed 
             offset[1] = 0;
             stride[0]= 1;
             stride[1] = 1;
             block[0]= 1; //each block of 2 values represents one complex number
             block[1] = 1;
             count[0]= signal_length*2;
             count[1] = 1;

             printf("Before memory hyperslab selection  \n");
             printf("offset is %d , %d  \n", (int)offset[0],(int)offset[1] );
	     printf("stride is   %d , %d  \n",(int)stride[0],(int)stride[1]);
	     printf("process %d count is   %d , %d  \n",mpi_rank, (int)count[0],(int)count[1]);
             printf("block is   %d , %d  \n",(int)block[0],(int)block[1]);
             status = H5Sselect_hyperslab(memspace_id,H5S_SELECT_SET,offset,NULL,count,block);

             
             
      
              printf("Before write operation \n");
             /* write the dataset. */
             start = clock();
             status = H5Dwrite(dataset_id, H5T_IEEE_F64LE, memspace_id, dataspace_id, plist2_id, multiplied_fft);
             end = clock();
             elapsed_time = (end-start)/CLOCKS_PER_SEC;
             //sprintf(perf_out+strlen(perf_out),",%f", elapsed_time);
             printf("Process %d : Data writing time is  %f seconds \n", mpi_rank, elapsed_time); 
             H5Dclose(dataset_id);
     
}
     H5Pclose(plist2_id);
     H5Sclose(dataspace_id);   
     H5Sclose(memspace_id);
     H5Fclose(file2_id);

    printf("Write function returning \n");

}
