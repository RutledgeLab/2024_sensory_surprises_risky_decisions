function [pval_sigstars,fontsize] = sigstar(pval)

pval_sigstars = {};
fontsize = NaN(length(pval),1);

    for x = 1:length(pval)
        if pval(x) < 1e-3
            txt = '***';
            fontsize(x) = 24;
        elseif pval(x) < 1e-2
            txt = '**';
            fontsize(x) = 24;
        elseif pval(x) < 0.05
            txt = '*';
            fontsize(x) = 24;
        elseif ~isnan(pval),
            % this should be smaller
            txt = 'n.s.';
            %txt = '';
            fontsize(x) = 16;
        else
            return
        end

        pval_sigstars{x,1} = txt;

    end
end
