
library(data.table)
library(ggplot2); theme_set(theme_bw())
library(stringr)
library(purrr)
library(magrittr)

PLOTS = TRUE

resdir <- here::here('results', 'efficiency')
figdir <- resdir
# figdir <- "~/git/fixedtx/writing/manuscript/figs/"

simprmdf <- fread(file.path(resdir, 'simprm_df.csv'))
simprmdf[, simprmidx:=copy(simprmsidx)]

result_files = list.files(resdir)
result_files = str_subset(result_files, "^n(\\d+)")

dfs <- map(result_files, ~fread(here::here(resdir, .x)))
names(dfs) <- result_files
rdf <- rbindlist(dfs, idcol='filename')
rdf[, `:=`(
  n = as.integer(str_extract(filename, "(?<=n)(\\d+)"))
)]

# merge in ground truth
rdf[simprmdf, `:=`(delta0_gt=i.delta0, delta1_gt=i.delta1), on='simprmidx']
rdf

rdf[, `:=`(error0=delta0 - delta0_gt, error1=delta1-delta1_gt)]
# rdfm <- melt(rdf, idvars = c('n', 'simprmidx', 'method', 'delta0_gt', 'delta1_gt'),
#              measure.vars = c('error0', 'error1'))
rdfm <- melt(rdf, idvars = c('n', 'simprmidx', 'method', 'delta0_gt', 'delta1_gt'),
             measure.vars = c('delta0', 'delta1'))
rdfm

rdfa <- rdfm[, list(sd=sd(value),
                    # bias=mean(value),
                    # ci_lo=quantile(value, probs=.025),
                    # ci_hi=quantile(value, probs=.975)),
                    ci_lo=quantile(value, probs=.025, na.rm=T),
                    ci_hi=quantile(value, probs=.975, na.rm=T)),
             by=c('n', 'simprmidx', 'estimator', 'variable')]
rdfa <- rdfa[simprmdf, on='simprmidx']
rdfa[, `:=`(
  # variable=factor(variable, levels=c('delta0', 'delta1'), labels=c('CATE(X=0)', 'CATE(X=1)')),
  estimand=factor(variable, levels=c('delta0', 'delta1'), labels=c('CATE(X=0)', 'CATE(X=1)')),
  method=copy(estimator)
  # estimator=copy(method)
)]
rdfa[, `:=`(ci_len=ci_hi-ci_lo)]

if (PLOTS) {
  color_map = c(
    'constrained' = 'darkgreen',
    'unconstrained' = 'orange'
  )
	# plot
	# ggplot(rdfa, aes(x=n, y=sd, col=estimator)) +
	#   geom_point() + geom_line(aes(linetype=estimand)) +
	#   geom_hline(aes(yintercept=0)) +
	#   facet_grid(gxt ~ b0+bx, labeller='label_both')

	# ggplot(rdfa, aes(x=n, y=sd, col=estimator)) + 
	#   geom_point() + geom_line(aes(linetype=estimand)) + 
	#   scale_x_log10() +
	#   scale_y_log10() +
	#   # geom_smooth(method='lm', se=F) +
	#   facet_grid(gxt ~ b0+bx, labeller='label_both')

	ggplot(rdfa, aes(x=n, y=ci_len, col=estimator)) +
	  geom_point() + geom_line(aes(linetype=estimand)) +
	  geom_hline(aes(yintercept=0), alpha=0) +
	  facet_grid(gxt ~ b0+bx, labeller='label_both') + 
    scale_color_manual(breaks=names(color_map), values=color_map) +
	  ggtitle("Length of confidence bounds versus sample size")

	width = 13.4; ratio = 1.414; height=width*ratio
	ggsave(file.path(figdir, 'ci_len_vs_n.pdf'),
	       width=width, height=height,
	       dpi=320)

	ggplot(rdfa, aes(x=n, y=ci_len, col=estimator)) + 
	  geom_point() + geom_line(aes(linetype=estimand)) + 
	  scale_x_log10() +
	  scale_y_log10() +
	  # geom_smooth(method='lm', se=F) +
	  facet_grid(gxt ~ b0+bx, labeller='label_both') +
    scale_color_manual(breaks=names(color_map), values=color_map) +
	  ggtitle("Log sample size versus log length of confidence bound")

	width = 13.4; ratio = 1.414; height=width*ratio
	ggsave(file.path(figdir, 'log_ci_len_vs_n.pdf'),
	       width=width, height=height,
	       dpi=320)


	# ggplot(rdfa, aes(x=ci_len, y=n, col=method)) + 
	#   geom_point() + geom_line(aes(linetype=estimand)) + 
	#   scale_x_log10() +
	#   scale_y_log10() +
	#   # geom_smooth(method='lm', se=F) +
	#   facet_grid(gxt ~ b0+bx, labeller='label_both')
}

# fit models
rdfa[, `:=`(log_n=log(n), log_sd=log(sd), log_ci_len=log(ci_len))]
fwrite(rdfa, file.path(resdir, 'agg.csv'), row.names=F)

# make average
rdfaw <- dcast(rdfa, simprmidx+log_n+method~variable, value.var = 'log_ci_len')
rdfaw[, log_ci_len:=(delta0+delta1)/2]
# fitdf <- rdfa[, list(fit=list(lm(log_ci_len~log_n+method))), by=c('simprmidx', 'variable')]
# fitdf <- rdfa[, list(fit=list(lm(log_ci_len~log_n+method*variable))), by=c('simprmidx')]
fitdf <- rdfaw[, list(fit=list(lm(log_n~log_ci_len+method))), by=c('simprmidx')]
# fitdf <- rdfa[, list(fit=list(lm(log_n~log_ci_len+method*variable))), by=c('simprmidx')]
fit1 <- fitdf$fit[[1]]
summary(fit1)
fitdf[, fit_coef:=map(fit, coef)]
fitdf[, fit_summary:=map(fit, summary)]
fitdf[, r_squared:=map_dbl(fit_summary, 'r.squared')]
fitdf[, `:=`(
  # b_log_n=map_dbl(fit_coef, 'log_n'),
  b_log_ci_len=map_dbl(fit_coef, 'log_ci_len'),
  b_m=map_dbl(fit_coef, 'methodunconstrained')
)]
fitdf[, `:=`(
  # frac_extra_needed = exp(-b_m/b_log_n) - 1
  frac_extra_needed = exp(b_m) - 1
)]
# print(fitdf[, list(frac_extra_needed_min=min(frac_extra_needed), frac_extra_needed_max=max(frac_extra_needed)), by='variable'])
# fitdf[, mean(b_log_n)]
# fitdf[, mean(b_m)]
# fitdf[, list(frac_extra_needed=exp(mean(-b_m) / mean(b_log_n)) - 1), by='variable']

# worst_fracs <- fitdf[, list(frac_extra_needed=max(frac_extra_needed)), by=c('simprmidx')]
# worst_fracs[, summary(frac_extra_needed)]
print(fitdf[, summary(frac_extra_needed)])

print(fitdf[, min(r_squared)])
fitdf[, plot(ecdf(r_squared))]
print(fitdf[, mean(r_squared>.95)])
