# An example of data prepropossing for UniTCR

## Requirements
R == 3.6.1  
Seurat == 3.2.2  
hdf5r == 1.3.3  
loomR == 0.2.1.9000  
We also run this code on R 4.1.0 with Seurat == 4.3.0.  

Python == 3.7.16  
scanpy == 1.9.3  
loompy == 3.0.7  

###  1.  Load the required packages
```{r}
library(Seurat)
library(hdf5r)
library(loomR)
```
### 2.  Read single-cell expression data to construct the Seurat object and perform initially filtering. The data utilized here is stored in the h5 format, hence the Read10X_h5 function is employed for reading. If your data is stored in other formats or has been encapsulated as Seurat objects, you can use the corresponding method to load it. In this scenario, two samples are sequentially read in.
```{r}
p1<-Read10X_h5("example_data/GSM6690564_nonnaiveCD8_post_pembro1_filtered_feature_bc_matrix.h5")
p1<-CreateSeuratObject(p1,project = "p1",min.cells = 3, min.features = 200)
p1[["percent.mt"]] <- PercentageFeatureSet(object = p1, pattern = "^MT-")
p1 <- subset(x = p1, subset = percent.mt < 10)  
p1

p2<-Read10X_h5("example_data/GSM6690566_nonnaiveCD8_post_pembro2_filtered_feature_bc_matrix.h5")
p2<-CreateSeuratObject(p2,project = "p2",min.cells = 3, min.features = 200)
p2[["percent.mt"]] <- PercentageFeatureSet(object = p2, pattern = "^MT-")
p2 <- subset(x = p2, subset = percent.mt < 10) 
p2
```

### 3.  Read the TCR sequence data corresponding to the sample. Select the appropriate method for reading in accordance with the file format. Each row of the data represents a cell, and each column corresponds to the TCR information of the cell. The barcode column corresponds to the cell's barcode in the expression profile, the chain column indicates whether the line represents an alpha chain or a beta chain, and the cdr3 column contains the amino acid sequence of the CDR3 region of the respective chain. These three columns of information are mandatory.

```{r}
p1TCR<-read.csv(gzfile("example_data/GSM6690565_nonnaiveCD8_post_pembro1_filtered_contig_annotations.csv.gz"))
head(p1TCR)

p2TCR<-read.csv(gzfile("example_data/GSM6690567_nonnaiveCD8_post_pembro2_filtered_contig_annotations.csv.gz"))

```

### 4.  Refine the function to filter TCR sequence data based on the number of beta chain detected in each cell,as well as its amino acid composition and length.

```{r}
###2.1 p1's TCR filtering----
TCR_filtering<-function(TCR_data){
  TCR_data<-TCR_data[TCR_data$chain == "TRB",]
  TCR_data<-TCR_data[,match(c("barcode","cdr3"),colnames(TCR_data))]

  ###Filter the cells that detect two or more beta chains
  dupBarcode<-(duplicated(TCR_data$barcode)|duplicated(TCR_data$barcode,fromLast = T))
  TCR_data<-TCR_data[!dupBarcode,]
  colnames(TCR_data)<-c("Barcode","Beta")
  TCR_data<-as.data.frame(TCR_data)
  TCR_data$Barcode<-as.character(TCR_data$Barcode)
  TCR_data$Beta<-as.character(TCR_data$Beta)

  ###Filter the cells based on the amino acid composition and length of the TCR sequence
  betaInd <- sapply(TCR_data$Beta, function(x) grepl("^[ACDEFGHIKLMNPQRSTVWY]+$", x))
  TCR_data<-TCR_data[betaInd,]
  
  bbbLength<-sapply(as.character(TCR_data$Beta),function(x){
    length(unlist(strsplit(x,split = "")))
  })
  lengthInd <- bbbLength >= 8 & bbbLength <= 25
  TCR_data<-TCR_data[lengthInd,]

  return(TCR_data)
}

```

### 5.  Use the TCR_filtering function to filter TCRs.

```{r}
p1TCR_filter<-TCR_filtering(p1TCR)
p2TCR_filter<-TCR_filtering(p2TCR)
head(p1TCR_filter)
head(p2TCR_filter)
```

### 6.  Combine the single cell expression profile data and TCR sequence information.

```{r}
interBarcode_1<-intersect(colnames(p1),p1TCR_filter$Barcode)
p1<-p1[,interBarcode_1]
p1TCR_filter<-p1TCR_filter[match(interBarcode_1,p1TCR_filter$Barcode),]
p1<-AddMetaData(p1,p1TCR_filter$Beta,col.name = "beta")
p1

interBarcode_2<-intersect(colnames(p2),p2TCR_filter$Barcode)
p2<-p2[,interBarcode_2]
p2TCR_filter<-p2TCR_filter[match(interBarcode_2,p2TCR_filter$Barcode),]
p2<-AddMetaData(p2,p2TCR_filter$Beta,col.name = "beta")
p2
```

### 7.  For each set of single-cell expression profiles, identify the highly variable genes. The number of highly variable genes is set to 5000 for all single-cell processes in this paper. Since this example involves two sets of data, the highly variable genes are identified separately for each set. If only one set of data is available, then the NormalizeData() and FindVariableFeatures() functions are used directly for identification.

```{r}
ifnb.list <- list(p1,p2)

ifnb.list <- lapply(X = ifnb.list, FUN = function(x) {
  x <- NormalizeData(x)
  x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 5000)
})
```

### 8.  Batch effect correction. This step can be omitted if the data shows no signs of batch effects. Proceed directly to subsequent standardization procedures.

```{r}
features <- SelectIntegrationFeatures(object.list = ifnb.list,nfeatures=5000)
intergrationAnchors <- FindIntegrationAnchors(object.list = ifnb.list, anchor.features = features)
intergrationSeurat <- IntegrateData(anchorset = intergrationAnchors)

intergrationSeurat
```

### 9.  Normalization of single-cell data.

```{r}
intergrationSeurat <- ScaleData(intergrationSeurat)
intergrationSeurat <- RunPCA(intergrationSeurat, npcs = 30)
intergrationSeurat <- RunUMAP(intergrationSeurat, reduction = "pca", dims = 1:30)
intergrationSeurat <- FindNeighbors(intergrationSeurat, reduction = "pca", dims = 1:30)
intergrationSeurat <- FindClusters(intergrationSeurat, resolution = 0.5)
```

### 10. As the subsequent modeling relies on Python, the processed data needs to be converted to the h5ad format. In this process, we utilize the loom format as an intermediary. First, the data in Seurat format is converted to loom format, and then from loom to h5ad. If you're working in a Python environment for data processing, you can directly save the results as h5ad format, by passing this step. It's important to note that during the conversion from Seurat to loom format, we transfer the expression data of highly variable genes after scaling.

```{r}
intergrationSeurat_loom<-intergrationSeurat[intergrationSeurat@assays[["integrated"]]@var.features,]
scale.data <- intergrationSeurat_loom@assays[["integrated"]]@scale.data
intergrationSeurat_loom <- SetAssayData(intergrationSeurat_loom,slot = 'data',new.data = scale.data,assay = "integrated")

sdata.loom <- as.loom(x = intergrationSeurat_loom, filename = "example_data/example.loom", verbose = FALSE)
sdata.loom$close_all()
```

## 11. To convert from loom format to h5ad format. This step is performed under Python.

```{python}
import scanpy as sc
adatas = sc.read_loom("example_data/example.loom", sparse=True, cleanup=False, dtype='float32')
adatas
adatas.write('example_data/example.h5ad')
