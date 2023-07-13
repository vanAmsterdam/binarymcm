# plot results of collapsibility experiment
library(data.table)
library(ggplot2); theme_set(theme_bw())
library(stringr)
library(purrr)
library(magrittr)

resdir <- here::here("results", "collapsibility")
figdir <- resdir
# figdir <- "~/git/fixedtx/writing/manuscript/figs/"

color_map = c(
  'offset' = 'orange',
  'CR-MCM' = 'darkgreen',
  'ATE' = 'black'
)

rdf <- fread(file.path(resdir, 'settingdf.csv'))
rdfm <- melt(rdf, measure.vars = c('offset', 'constrained', 'ate'),
             value.name = 'pehe2',
             variable.name = 'estimator')
rdfm[, pehe:=sqrt(pehe2)]

rdfm[estimator=='ate', estimator:='ATE']
rdfm[estimator=='constrained', estimator:='CR-MCM']

ggplot(rdfm, aes(x=bx, y=pehe, col=estimator)) + 
  geom_point() + geom_line() + 
  scale_color_manual(breaks=names(color_map), values=color_map) +
  labs(y='PEHE', x=expression(beta[x])) + 
  guides(color=guide_legend(title='model'))

width=20
ratio=2.5
height=width / ratio

ggsave(file.path(figdir, 'collapsibility.pdf'),
       width=width, height=height, units='cm', dpi=300)
