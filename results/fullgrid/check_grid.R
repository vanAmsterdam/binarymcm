# plot results of confounding experiment
suppressPackageStartupMessages({
	library(data.table)
	library(stringr)
	library(purrr)
  library(ggplot2); theme_set(theme_bw())
  library(ggh4x)
  library(xtable)
})

color_map = c(
  `full` = 'red',
  'offset' = 'orange',
  'constrained' = 'darkgreen',
  'ATE' = 'black'
)

expdir = here::here('results', 'fullgrid')
figdir <- "~/git/fixedtx/writing/manuscript/figs"
figdir = file.path(expdir, 'figs')
if (!file.exists(figdir)) mkdirs(figdir)

result_files = list.files(expdir)
result_files = str_subset(result_files, '^settingdf')


rdfs <- map(result_files, ~fread(file.path(expdir, .x)))
names(rdfs) <- result_files
rdf <- rbindlist(rdfs, idcol='filename')
# rdf <- fread('results/fullgrid/settingdf27.csv')

# check duplicates
rescols <- c('ate', 'offset', 'constrained3', 'constrained4',
             'll_full', 'll_tx', 'll_tu', 'q_min', 'q_max', 'btx')
admincols <- c('filename', 'settingidx')
prmcols <- setdiff(colnames(rdf), c(rescols, admincols))
rdfu <- unique(rdf, by=prmcols, fromLast = T)

## dummy calculations
# d0 = .15; d1 = .16
# dd = d1 - d0
# px = 0.5
# dbar = (1-px)*d0+px*d1
# var_cate = (1-px)*(d0 - dbar)^2 + px * (d1 - dbar)^2
# var_cate - 0.25 * dd^2

# make the criteria
rdfu[, `:=`(
  offset_better=offset<ate,
  constrained3_better=constrained3<ate,
  constrained4_better=constrained4<ate,
  constrained3_better_than_offset=constrained3<offset,
  constrained4_better_than_offset=constrained4<offset,
  constrained3_better_than_eq_offset=constrained3<=offset,
  constrained4_better_than_eq_offset=constrained4<=offset,
  best_pehe=pmin(ate, offset, constrained3, constrained4),
  ddelta_gt0=ate> 0,
  ddelta_gt1=ate> 0.25 * 0.01^2,
  ddelta_gt5=ate> 0.25 * 0.05^2,
  # var_cate_gt0=ate>0,
  # sd_cate_gt05=ate>0.05^2,
  offset_fulfilled=abs(btx) < 1e-3,
  avoids_symmetrypoint = q_max < 0.5 | q_min > 0.5,
  lltx_gt_lltu = ll_tx < ll_tu # note its nll that's returned in the experiments
)]
contenders <- c('ate', 'offset', 'constrained3', 'constrained4')
walk(contenders, ~set(rdfu, j=paste0(.x,'_best'), value=rdfu$best_pehe == rdfu[[.x]]))
crit_vars_base <- c('offset_fulfilled', 'avoids_symmetrypoint', 'lltx_gt_lltu') 
crit_vars0 <- c('ddelta_gt0', crit_vars_base)
crit_vars1 <- c('ddelta_gt1', crit_vars_base)
crit_vars5 <- c('ddelta_gt5', crit_vars_base)

fwrite(rdfu, file.path(expdir, 'nondup_results.csv'), row.names=F)

# by delta constraint
rdfa0 <- rdfu[, list(n_exps = .N,
                     frac_offset_better=mean(offset_better),
                     frac_constrained3_better=mean(constrained3_better),
                     frac_constrained4_better=mean(constrained4_better),
                     frac_constrained3_better_than_offset=mean(constrained3_better_than_offset),
                     frac_constrained4_better_than_offset=mean(constrained4_better_than_offset),
                     frac_constrained3_better_than_eq_offset=mean(constrained3_better_than_eq_offset),
                     frac_constrained4_better_than_eq_offset=mean(constrained4_better_than_eq_offset),
                     frac_ate_best=mean(ate_best),
                     frac_offset_best=mean(offset_best),
                     frac_constrained3_best=mean(constrained3_best),
                     frac_constrained4_best=mean(constrained4_best)
                     ), by=crit_vars0]
rdfa1 <- rdfu[, list(n_exps = .N,
                     frac_offset_better=mean(offset_better),
                     frac_constrained3_better=mean(constrained3_better),
                     frac_constrained4_better=mean(constrained4_better),
                     frac_constrained3_better_than_offset=mean(constrained3_better_than_offset),
                     frac_constrained4_better_than_offset=mean(constrained4_better_than_offset),
                     frac_constrained3_better_than_eq_offset=mean(constrained3_better_than_eq_offset),
                     frac_constrained4_better_than_eq_offset=mean(constrained4_better_than_eq_offset),
                     frac_ate_best=mean(ate_best),
                     frac_offset_best=mean(offset_best),
                     frac_constrained3_best=mean(constrained3_best),
                     frac_constrained4_best=mean(constrained4_best)
                     ), by=crit_vars1]
rdfa5 <- rdfu[, list(n_exps = .N,
                     frac_offset_better=mean(offset_better),
                     frac_constrained3_better=mean(constrained3_better),
                     frac_constrained4_better=mean(constrained4_better),
                     frac_constrained3_better_than_offset=mean(constrained3_better_than_offset),
                     frac_constrained4_better_than_offset=mean(constrained4_better_than_offset),
                     frac_constrained3_better_than_eq_offset=mean(constrained3_better_than_eq_offset),
                     frac_constrained4_better_than_eq_offset=mean(constrained4_better_than_eq_offset),
                     frac_ate_best=mean(ate_best),
                     frac_offset_best=mean(offset_best),
                     frac_constrained3_best=mean(constrained3_best),
                     frac_constrained4_best=mean(constrained4_best)
                     ), by=crit_vars5]
rdfas <- list(`0`=rdfa0, `1`=rdfa1, `5`=rdfa5)
walk(rdfas, setnames, c('ddelta_gt0', 'ddelta_gt1', 'ddelta_gt5'), c('ddelta_crit', 'ddelta_crit', 'ddelta_crit'),
    skip_absent=T)
rdfa <- rbindlist(rdfas, idcol='ddelta_min')
rdfa[, `:=`(ddelta_min=as.numeric(ddelta_min) / 100)]
fwrite(rdfa, file.path(expdir, 'aggresults_4crit.csv'), row.names=F)

# condense by 
rdfaa1 <- rdfa[, list(
  frac_offset_better=sum(frac_offset_better*n_exps)/sum(n_exps),
  frac_constrained3_better=sum(frac_constrained3_better*n_exps)/sum(n_exps),
  frac_constrained4_better=sum(frac_constrained4_better*n_exps)/sum(n_exps),
  frac_constrained3_better_than_offset=sum(frac_constrained3_better_than_offset*n_exps)/sum(n_exps),
  frac_constrained4_better_than_offset=sum(frac_constrained4_better_than_offset*n_exps)/sum(n_exps),
  frac_constrained3_better_than_eq_offset=sum(frac_constrained3_better_than_eq_offset*n_exps)/sum(n_exps),
  frac_constrained4_better_than_eq_offset=sum(frac_constrained4_better_than_eq_offset*n_exps)/sum(n_exps),
  frac_ate_best=sum(frac_ate_best*n_exps)/sum(n_exps),
  frac_offset_best=sum(frac_offset_best*n_exps)/sum(n_exps),
  frac_constrained3_best=sum(frac_constrained3_best*n_exps)/sum(n_exps),
  frac_constrained4_best=sum(frac_constrained4_best*n_exps)/sum(n_exps),
  n_exps=sum(n_exps)),
  by=c('ddelta_min', 'ddelta_crit')]
rdfaa2 <- rdfa[, list(
  frac_offset_better=sum(frac_offset_better*n_exps)/sum(n_exps),
  frac_constrained3_better=sum(frac_constrained3_better*n_exps)/sum(n_exps),
  frac_constrained4_better=sum(frac_constrained4_better*n_exps)/sum(n_exps),
  frac_constrained3_better_than_offset=sum(frac_constrained3_better_than_offset*n_exps)/sum(n_exps),
  frac_constrained4_better_than_offset=sum(frac_constrained4_better_than_offset*n_exps)/sum(n_exps),
  frac_constrained3_better_than_eq_offset=sum(frac_constrained3_better_than_eq_offset*n_exps)/sum(n_exps),
  frac_constrained4_better_than_eq_offset=sum(frac_constrained4_better_than_eq_offset*n_exps)/sum(n_exps),
  frac_ate_best=sum(frac_ate_best*n_exps)/sum(n_exps),
  frac_offset_best=sum(frac_offset_best*n_exps)/sum(n_exps),
  frac_constrained3_best=sum(frac_constrained3_best*n_exps)/sum(n_exps),
  frac_constrained4_best=sum(frac_constrained4_best*n_exps)/sum(n_exps),
  n_exps=sum(n_exps)),
  # by=c('offset_fulfilled', 'ddelta_min', 'ddelta_crit')]
  by=c('ddelta_min', 'ddelta_crit', 'offset_fulfilled')]
# setorder(rdfa2, offset_fulfilled, ddelta_min, ddelta_crit)

# prep for printing
setorder(rdfaa1, ddelta_min, ddelta_crit)
#printtable1 <- rdfaa1[ddelta_crit==T, .SD, .SDcols=setdiff(colnames(rdfaa1), 'ddelta_crit')]
printtable1 <- rdfaa1[ddelta_crit==T, .SD, .SDcols=c('ddelta_min', 'frac_offset_better', 'frac_constrained3_better', 'frac_constrained4_better', 'n_exps')]
setnames(printtable1, 
         c('ddelta_min', 'frac_offset_better', 'frac_constrained3_better', 'frac_constrained4_better', 'n_exps'),
         c('$\\delta$', 'offset', 'CR-MCM', 'MCM', 'N')
         )
xt1 <- xtable(printtable1, digits=3)
print(xt1, file=file.path(figdir, 'fullgrid_1crit.tex'), include.rownames=F)



setorder(rdfaa2, ddelta_min, ddelta_crit, offset_fulfilled)
#printtable2 <- rdfaa2[ddelta_crit==T, .SD, .SDcols=setdiff(colnames(rdfaa2), 'ddelta_crit')]
printtable2 <- rdfaa2[ddelta_crit==T, .SD, .SDcols=c('ddelta_min', 'offset_fulfilled', 'frac_offset_better', 'frac_constrained3_better', 'frac_constrained4_better', 'n_exps')]

setnames(printtable2, 
         c('ddelta_min', 'offset_fulfilled', 'frac_offset_better', 'frac_constrained3_better', 'frac_constrained4_better', 'n_exps'),
         c('delta', 'offset satisfied', 'offset', 'CR-MCM', 'MCM', 'N')
         )
# latextable2 <- kableExtra::kbl(printtable2, booktabs=T)
# kableExtra::save_kable(latextable2, file.path(figdir, 'fullgrid_2crit.tex'))
# library(stargazer)
# stargazer(printtable2, title="Results of grid experiment", out = file.path(figdir, 'fullgrid_2crit.tex'))
xt2 <- xtable(printtable2)
print(xt2, file=file.path(figdir, 'fullgrid_2crit.tex'), include.rownames=F)


fwrite(rdfaa1, file.path(expdir, 'aggresults_1crit.csv'), row.names=F)
fwrite(rdfaa2, file.path(expdir, 'aggresults_2crit.csv'), row.names=F)

## make plots
# rdfa[, figlabel:=paste0(round(100*frac_constrained_better), '% N experiments=', n_exps)]
# g0 <- ggplot(rdfa[ddelta_min==0.], aes(x=1,y=1)) + 
#   geom_tile(aes(fill=frac_constrained_better), alpha=0.75) +
#   geom_text(aes(label=figlabel)) +
#   facet_nested(ddelta_crit+offset_fulfilled~lltx_gt_lltu+avoids_symmetrypoint,
#              labeller='label_both') + 
#   scale_fill_viridis_c() + 
#   labs(x='', y='') + 
#   theme(
#     axis.text.x = element_blank(),
#     axis.text.y = element_blank(),
#     axis.ticks = element_blank(),
#     panel.spacing = unit(0, 'lines'),
#     panel.grid.major = element_blank(),
#     panel.grid.minor = element_blank()
#     ) + 
#   ggtitle("fraction of experiments in which the marginally constrained offset estimator has lower PEHE than the ATE-baseline",
#           paste0("total number of experiments: ", nrow(rdfu)))
# g1 = g0 %+% rdfa[ddelta_min==.01]
# g5 = g0 %+% rdfa[ddelta_min==.05]
# ggsave(file.path(figdir, 'fullgrid_ddelta0.png'), plot=g0, width=13.4, height=6.14)
# ggsave(file.path(figdir, 'fullgrid_ddelta1.png'), plot=g1, width=13.4, height=6.14)
# ggsave(file.path(figdir, 'fullgrid_ddelta5.png'), plot=g5, width=13.4, height=6.14)

print(paste0(nrow(rdfu), ' unique experiments, removed ', nrow(rdf) - nrow(rdfu), ' duplicates'))

#print(rdfaa)

## density plot
#ggplot(rdfu, aes(x=offset, y=constrained3)) + 
  #geom_hex(bins=35) + 
  #scale_fill_continuous(type='viridis') + 
  #facet_grid(offset_fulfilled~ddelta_gt5)


