import tmscoring

def get_scores(path_to_file_one, path_to_file_two): 
    # return tm score and rmsd score 
    alignment = tmscoring.TMscoring(path_to_file_one, path_to_file_two)
    alignment.optimise()
    return (alignment.tmscore(**alignment.get_current_values()), alignment.rmsd(**alignment.get_current_values())) 
