* Preprocessing of data services
    * AWS Glue,
    * Amazon EMR, 
    * Amazon Redshift
    * Amazon Relational Database Service
    * Amazon Athena.

* which is the best options to convert input data to recordio format ?
    * Glue cannot write to recordIO-protobuf format

* pipe mode is applicable to both csv and recordio file formats

* different channels.
    * The concept of channels is the same as the concept of "Datastores" in azure. The data resides in the storage solution (s3 in most cases) and channels are the logical path to your data. Only during training,vaidation, testing and for storing model artifacts do we requires to have access to them.
    
    * Input Data channel
        * Format
            * ChannelName:[train/test/validation] This is a required parameter
            * DataSource: [FileSystemDataSource/S3DataSource]This is a required parameter
            * ContentType
            * CompressionType : Used only in pipe mode
            * InputMode: [File/Pipe]
        * Example:
            "InputDataConfig": 
            [
                {
                    "ChannelName": "train",
                    "DataSource": {
                        "S3DataSource": {
                                            "S3DataType": "S3Prefix",
                                            "S3Uri": f"{bucket_path}/{prefix}/train/",
                                            "S3DataDistributionType": "FullyReplicated",
                                        }
                                    },
                    "ContentType": "libsvm",
                    "CompressionType": "None",
                },
                {
                    "ChannelName": "validation",
                    "DataSource": {
                        "S3DataSource": {
                                            "S3DataType": "S3Prefix",
                                            "S3Uri": f"{bucket_path}/{prefix}/validation/",
                                            "S3DataDistributionType": "FullyReplicated",
                                        }
                    },
                    "ContentType": "libsvm",
                    "CompressionType": "None",
                },
            ],







    * Validation:
        * predictor.predict(array, initial_args={"ContentType": "application/json"})


## Tips
* While deciding on algorithm choice or on dataPre-processing/ETL solution choice use the information of which algorithm supports which type of data format as input. 
    * For example: 
        * IP Insights algorithm supports only CSV file type as training data, so other options using parquet or recordIO-protobuf are ruled out.
        * Glue cannot write output in recordIO-protobuf format.

## Random Information
* AWS Glue job cannot write output in recordIO-protobuf format.

* Object2Vec can be used to find semantically similar objects such as tickets. BlazingText Word2Vec can only find semantically similar words.

* The Training Image and Inference Image Registry Paths are region based (geographically).

* Only files and data saved within the /home/ec2-user/SageMaker folder persist between notebook instance sessions.

* Amazon SageMaker models are stored as model.tar.gz in the S3 bucket specified in OutputDataConfig S3OutputPath parameter of the create_training_job call.

* Within an inference pipeline model, Amazon SageMaker handles invocations as a sequence of HTTP requests.https://docs.aws.amazon.com/sagemaker/latest/dg/inference-pipelines.html

* Early Stopping is decided by median. https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-early-stopping.html
    * If the value of the objective metric for the current training job is worse (higher when minimizing or lower when maximizing the objective metric) than the median value of running averages of the objective metric for previous training jobs up to the same epoch, Amazon SageMaker stops the current training job.

* If you disable direct internet access, the notebook instance won't be able to train or host models unless your VPC has an interface endpoint (PrivateLink) or a NAT gateway and your security groups allow outbound connections. Please read more: https://docs.aws.amazon.com/sagemaker/latest/dg/appendix-notebook-and-internet-access.html


* Kinesis Data Analytics cannot directly ingest source data.
* Kinesis Firehose with lambda would introduce a buffering delay of at least 1 minute (Its near real-time, for real time applications use Kinesis Data Streams, or Kinesis Video Stream (for video))


* Use categorical binning instead of One-Hot encoding when the categories have order.



## Overall Process

* Data Injestion

* Data Transformation

* Training

    * CREATE/FETCH REQUIRED RESOURCES:
        * **output_path**:The URL of the Amazon Simple Storage Service (Amazon S3) bucket where you've stored the training data.
        * **train_instance_type**:The compute resources that you want SageMaker to use for model training. Compute resources are ML compute instances that are  managed by SageMaker.
        * **output_path**: The URL of the S3 bucket where you want to store the output of the job.
        * **image**:The Amazon Elastic Container Registry path where the training code is stored.
    
    
    * CREATE AN ESTIMATOR / CHOOSE AN ALGORITHM
        * Choosing an estimator
            * If Deep Learning
                * Use an algorithm provided by SageMaker
                * Use SageMaker Debugger (Tensoflow, PyTorch, ApacheMXNet)
                * Use Apache Spark with SageMaker
                * Submit custom code to train with deep learning frameworks
                * Use an algorithm that you subscribe to from AWS Marketplace
            * If not Deep Learning
                * Use an algorithm provided by SageMaker
                * Use SageMaker Debugger (XGBoost)
                * Use Apache Spark with SageMaker
                * Use your own custom algorithms : CreateTrainingJob API call
                * Use an algorithm that you subscribe to from AWS Marketplace
        * Parameters Required.
            https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html
            * role is a mandatory parameter.
            * sagemaker.estimator.EstimatorBase(role, instance_count=None, instance_type=None, volume_size=30, volume_kms_key=None, max_run=86400, input_mode='File', output_path=None, output_kms_key=None, base_job_name=None, sagemaker_session=None, tags=None, subnets=None, security_group_ids=None, model_uri=None, model_channel_name='model', metric_definitions=None, encrypt_inter_container_traffic=False, use_spot_instances=False, max_wait=None, checkpoint_s3_uri=None, checkpoint_local_path=None, rules=None, debugger_hook_config=None, tensorboard_output_config=None, enable_sagemaker_metrics=None, enable_network_isolation=False, profiler_config=None, disable_profiler=False, environment=None, max_retry_attempts=None, **kwargs)

            * OR

            * sagemaker.estimator.Estimator(image_uri, role, **)
                * generic Estimator
        
        * model.fit(inputs=None, wait=True, logs='All', job_name=None, experiment_config=None)
            * Train a model using the input training dataset.
        
        
        * model.compile_model(target_instance_family, input_shape, output_path, framework=None, framework_version=None, compile_max_run=900, tags=None,         target_platform_os=None, target_platform_arch=None, target_platform_accelerator=None, compiler_options=None, **kwargs)
            * Compile a Neo model using the input model.
        
        * logs()
            * Display logs.
        
        * register(content_types, response_types, inference_instances, transform_instances, image_uri=None, model_package_name=None, model_package_group_name=None, model_metrics=None, metadata_properties=None, marketplace_cert=False, approval_status=None, description=None, compile_model_family=None, model_name=None, **kwargs)
            * Creates a model package for creating SageMaker models or listing on Marketplace.

        
    * Note:
        * When you create a training job with the API, SageMaker replicates the entire dataset on ML compute instances by default. To make SageMaker replicate a subset of the data on each ML compute instance, you must set the S3DataDistributionType field to ShardedByS3Key. You can set this field using the low-level SDK. For more information, see S3DataDistributionType in S3DataSource.


* Validation
    * You can evaluate your model using historical data (offline) or live data:
        * Offline testing—Use historical, not live, data to send requests to the model for inferences.
            * Validating using a holdout set
            * k-fold validation
        * Online testing with live data—SageMaker supports A/B testing for models in production by using production variants. 
    


* Deployment
    * Deploying a model using SageMaker hosting services is a three-step process:
        1. Create a model in SageMaker:  CreateModel API.
            * The S3 bucket where the model artifacts are stored must be in the same region as the model that you are creating.

        2. Create an endpoint configuration for an HTTPS endpoint: CreateEndpointConfig API.
        
        3. Create an HTTPS endpoint: InvokeEndpoint API.

    * Real Time
        * create_model(role=None, image_uri=None, predictor_cls=None, vpc_config_override='VPC_CONFIG_DEFAULT', **kwargs)
            * Create a SageMaker Model object that can be deployed to an Endpoint.

    * Batch Transform
        * transformer(instance_count, instance_type, strategy=None, assemble_with=None, output_path=None, output_kms_key=None, accept=None, env=None, max_concurrent_transforms=None, max_payload=None, tags=None, role=None, volume_kms_key=None, vpc_config_override='VPC_CONFIG_DEFAULT', enable_network_isolation=None, model_name=None)

    * model.deploy(initial_instance_count, instance_type, serializer=None, deserializer=None, accelerator_type=None, endpoint_name=None, use_compiled_model=False, wait=True, model_name=None, kms_key=None, data_capture_config=None, tags=None, **kwargs)

* Monitoring

