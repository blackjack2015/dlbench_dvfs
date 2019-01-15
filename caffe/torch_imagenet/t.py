from datasets import DatasetHDF5    

    def imagenet_prepare(self):
        # Data loading code
        traindir = os.path.join(self.data_dir, 'train')
        testdir = os.path.join(self.data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        #trainset = torchvision.datasets.ImageFolder(traindir, transforms.Compose([
        hdf5fn = os.path.join(self.data_dir, 'imagenet-shuffled.hdf5')
        trainset = DatasetHDF5(hdf5fn, 'train', transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ]))
        self.trainset = trainset

        train_sampler = None
        shuffle = False
        if self.nworkers > 1: 
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.trainset, num_replicas=self.nworkers, rank=self.rank)
            train_sampler.set_epoch(0)
            shuffle = False
        self.train_sampler = train_sampler

        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batch_size, shuffle=shuffle,
            num_workers=NUM_CPU_THREADS, pin_memory=True, sampler=train_sampler)
        #testset = torchvision.datasets.ImageFolder(testdir, transforms.Compose([
        testset = DatasetHDF5(hdf5fn, 'val', transforms.Compose([
                transforms.ToPILImage(),
        #        transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        self.testset = testset
        self.testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.batch_size, shuffle=False,
            num_workers=1, pin_memory=True)

