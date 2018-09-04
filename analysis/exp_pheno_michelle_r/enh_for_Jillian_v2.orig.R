#load data
enhancer<-read.csv("enh_query.csv")
enhancerdf<- as.data.frame(enhancer)
library(dplyr)
library(tidyr)

#formatting and add calculated columns for 
enhancerdf$count_larva<-as.numeric(as.character(enhancerdf$count_larva))
enhancerdf$count_embryo<-as.numeric(as.character(enhancerdf$count_embryo))
enhancerdf$count_adult<-as.numeric(as.character(enhancerdf$count_adult))
enhancerdf$larvaperadult<- enhancerdf$count_larva / enhancerdf$count_adult
enhancerdf$broodsize<- enhancerdf$count_embryo + enhancerdf$count_larva
enhancerdf$survival<- enhancerdf$count_larva / (enhancerdf$count_embryo + enhancerdf$count_larva)
enhancerdf$date<-as.Date(as.character(enhancerdf$date))

#subset df to include only experiments done after Feb 1, 2015 (enhancer set)
enhancerdf <- subset(enhancerdf, date > "2015-02-01")

#remove NAS
enhancerdf<-enhancerdf %>%
  drop_na()

########

#just some data visualization

highsurvival <- subset(enhancerdf, survival>= .7)

lowsurvival <- subset(enhancerdf, survival< .7)

library(ggplot2)

enhden <- ggplot(lowsurvival, aes(x=survival)) + 
  geom_density() +
  xlab("Devstar Survival")+
  facet_wrap(~ gene)
enhden

enhden2 <- ggplot(enhancerdf, aes(x=survival)) + 
  geom_density() +
  xlab("Devstar Survival")+
  facet_wrap(~ gene)
enhden2 +ggtitle("Distribution of Devstar Enhancer Scores")

#######

##read and process N2 scores

enhancer_N2<-read.csv("enh_N2.csv")
enhancerdf_N2<- as.data.frame(enhancer_N2)

enhancerdf_N2$count_larva<-as.numeric(as.character(enhancerdf_N2$count_larva))
enhancerdf_N2$count_embryo<-as.numeric(as.character(enhancerdf_N2$count_embryo))
enhancerdf_N2$count_adult<-as.numeric(as.character(enhancerdf_N2$count_adult))
enhancerdf_N2$larvaperadult<- enhancerdf_N2$count_larva / enhancerdf_N2$count_adult
enhancerdf_N2$broodsize<- enhancerdf_N2$count_embryo + enhancerdf_N2$count_larva
enhancerdf_N2$survival<- enhancerdf_N2$count_larva / (enhancerdf_N2$count_embryo + enhancerdf_N2$count_larva)
enhancerdf_N2$date<-as.Date(as.character(enhancerdf_N2$date))

#subset df to include only experiments done after Feb 1, 2015 (enhancer set)
enhancerdf_N2 <- subset(enhancerdf_N2, date > "2015-02-01")

#remove NAS
enhancerdf_N2<-enhancerdf_N2 %>%
  drop_na()

##take mean of N2 x RNAi replicates
#enh_N2_mean<-ddply(enhancerdf_N2, .(gene, date, library_stock_id), summarize, mean=mean(survival))
enh_N2_mean <-enhancerdf_N2 %>%
  group_by(gene, date, library_stock_id) %>%
  summarize(mean_N2=mean(survival))
###
#more data visualization- distribution of devstar N2 scores

N2com <- ggplot(enh_N2_mean, aes(x=mean_N2)) + 
  geom_density() +
  xlab("Devstar Survival") +
  facet_wrap(~ date)
N2com +ggtitle("Distribution of Devstar N2 Scores")
###

##read and process TS x L44440 scores

enhL4440<-read.csv("enh_L4440.csv")
enhL4440df<- as.data.frame(enhL4440)

#formatting and add calculated columns  
enhL4440df$count_larva<-as.numeric(as.character(enhL4440df$count_larva))
enhL4440df$count_embryo<-as.numeric(as.character(enhL4440df$count_embryo))
enhL4440df$count_adult<-as.numeric(as.character(enhL4440df$count_adult))
enhL4440df$larvaperadult<- enhL4440df$count_larva / enhL4440df$count_adult
enhL4440df$broodsize<- enhL4440df$count_embryo + enhL4440df$count_larva
enhL4440df$survival<- enhL4440df$count_larva / (enhL4440df$count_embryo + enhL4440df$count_larva)
enhL4440df$date<-as.Date(as.character(enhL4440df$date))

#subset df to include only experiments done after Feb 1, 2015 (enhL4440 set)
enhL4440df <- subset(enhL4440df, date > "2015-02-01")

#more formatting
enhL4440df<- separate(enhL4440df, experiment_id, c("plate", "well"), "_")

#remove NAS
enhL4440df<-enhL4440df %>%
  drop_na()

###

#more data visualization
#distribution of devstar scores for TS x L4440  by query gene, date
enhL4440den2 <- ggplot(enhL4440df, aes(x=survival, color=as.character(date), group=plate)) + 
  geom_density() +
  facet_wrap(~ gene)
enhL4440den2 + ggtitle("Distribution of Devstar Scores for TS x L4440")

###
#calculate mean and standard deviation of survival of TSxL4440 by date

#enhL4440df_sum<-ddply(enhL4440df, .(gene, date), 
#                      summarize, mean=mean(survival), sd= sd(survival))
enhL4440df_sum <- enhL4440df %>%
  group_by(gene, date) %>%
  summarize(mean_survival_L4440=mean(survival), 
            sd_survival_L4440= sd(survival),
            mean_broodsize_L4440=mean(broodsize), 
            sd_broodsize_L4440= sd(broodsize))

#merge TSxRNAi and TSxL4440 scores
enh_L4440_merge <- merge(enhancerdf, enhL4440df_sum, by = c("date", "gene"))

#calculate z score (approxiamte)
enh_L4440_merge$z_survival <- (enh_L4440_merge$survival - 
                                 enh_L4440_merge$mean_survival_L4440)/enh_L4440_merge$sd_survival_L4440

#get percentile from z score
enh_L4440_merge$norm_survival <- pnorm(enh_L4440_merge$z_survival)

enh_L4440_merge$norm2_survival <- 1- enh_L4440_merge$norm_survival

###
#more data visualization- distribution of z score
enhL4440norm <- ggplot(enh_L4440_merge, aes(x=z_survival, color=as.character(date))) + 
  geom_density() +
  facet_wrap(~ gene)
enhL4440norm +ggtitle("Distribution of Z Scores")
###
#z score for broodsize (ste)
enh_L4440_merge$z_broodsize <- (enh_L4440_merge$broodsize - 
                                  enh_L4440_merge$mean_broodsize_L4440)/enh_L4440_merge$sd_broodsize_L4440

#get percentile from z score
enh_L4440_merge$norm_broodsize <- pnorm(enh_L4440_merge$z_broodsize)

enh_L4440_merge$norm2_broodsize <- 1- enh_L4440_merge$norm_broodsize

#merge all three data types
enh_L4440_N2_merge <- merge(enh_L4440_merge, enh_N2_mean, by = c("date", "library_stock_id"))

#select positives
enh_L4440_N2_pos <- subset(enh_L4440_N2_merge, 
                           (norm2_survival > .98 & mean_N2 > .5) | 
                             norm2_broodsize > .98 )

##summary of positives

enh_diff_l4440_N2 <- enh_L4440_N2_pos %>%
  group_by(gene.x, date, library_stock_id) %>%
  summarize(count=length(library_stock_id))

##take subset of positives where at least 3 (usually there are 8 total) replicates passed the threshold
#summary of the number of experiments for each query gene that pass
enh_diff_l4440_N2_a <- subset(enh_diff_l4440_N2, count > 2 ) %>%
  group_by(gene.x) %>%
  summarize(count=length(gene.x))




###
#more data visualization
enhL4440N2 <- ggplot(enh_L4440_N2_pos, aes(x=survival, color=as.character(date))) + 
  geom_density() +
  facet_wrap(~ gene.x)
enhL4440N2
###
###data visualization

graph <- enh_L4440_N2_merge %>%
  mutate(color = ifelse((norm2_survival > .98 & mean_N2 > .5), "A", "B")) %>%
  subset(color!="NA")

g1 <- ggplot(graph, aes(x=survival, fill=color)) + 
  geom_histogram(binwidth = 0.01, aes(alpha=.5)) +
  scale_fill_manual(values= c("#E69F00", "#56B4E9"))+
  theme(legend.title=element_blank(),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank()) +
  facet_wrap(~ gene.x)
g1

##to get gene lists for scoring program (emb-30 as an example)
emb30_score <- subset(subset(enh_diff_l4440_N2, gene.x == "emb-30" & count > 2 ), 
                      select= c("gene.x", "date", "library_stock_id"))
write.csv(emb30_score, file= "emb30_enh_for_scoring.csv")


