#!/usr/bin/env python
from featureExtractor import *
from labels import *

trainingLabels = {'data/agatha_christie.txt': 1,\
 'data/agnes_laut.txt': 1, 'data/alec_waugh.txt': -1, 'data/alfred_ollivant.txt': -1,\
 'data/alice_brown.txt': 1, 'data/alice_green.txt': 1, 'data/alice_hazeltine.txt': 1,\
 'data/alice_rice.txt': 1, 'data/allan_arnold.txt': -1, 'data/allan_pinkerton.txt': -1,\
 'data/allen_chapman.txt': -1, 'data/amanda_douglas.txt': 1, 'data/andrew_spedon.txt': -1,\
 'data/angela_brazil.txt': 1, 'data/ann_stephens.txt': 1, 'data/anna_green.txt': 1,\
 'data/annabel_sharp.txt': 1, 'data/anne-jane-couples.txt': 1, 'data/annie_besant.txt': 1,\
 'data/annie_donnell.txt': 1, 'data/annie_johnston.txt': 1, 'data/anthony_hope.txt': -1,\
 'data/anthony_trollope.txt': -1, 'data/anton_chekhov.txt': -1, 'data/arnold_bennett.txt': -1,\
 'data/arthur-conan-doyle.txt': -1, 'data/arthur_bailey.txt': -1, 'data/arthur_griffiths.txt': -1,\
 'data/arthur_harding.txt': -1, 'data/arthur_stringer.txt': -1, 'data/arthur_symons.txt': -1,\
 'data/arthur_waltermire.txt': -1, 'data/beatrice_baskerville.txt': 1, 'data/beatrice_harraden.txt': 1,\
 'data/beatrix_potter.txt': 1, 'data/berta_ruck.txt': 1, 'data/blanche_devereux.txt': 1,\
 'data/blanche_macdonnell.txt': 1, 'data/bram_stoker.txt': -1, 'data/capwell_wyckoff.txt': -1,\
 'data/caroline_wells.txt': 1, 'data/carolyn_wells.txt': 1, 'data/charles-dickens.txt': -1,\
 'data/charles_hawes.txt': -1, 'data/charles_hudson.txt': -1, 'data/charles_king.txt': -1,\
 'data/charles_kingsley.txt': -1, 'data/charles_morgan.txt': -1, 'data/charlotte-bronte.txt': 1,\
 'data/charlotte-eaton.txt': 1, 'data/charlotte_gilman.txt': 1, 'data/christine_beals.txt': 1,\
 'data/clara_corfield.txt': 1, 'data/clara_guernsey.txt': 1, 'data/clara_mullholland.txt': 1,\
 'data/clarence_cullen.txt': -1, 'data/compton_mackenzie.txt': -1, 'data/contance_woolson.txt': 1,\
 'data/cyril_burleigh.txt': -1, 'data/david_stafford.txt': -1, 'data/dinah_craik.txt': 1,\
 'data/dixon_scott.txt': -1, 'data/dora_russell.txt': 1, 'data/dorothy_richardson.txt': 1,\
 'data/eden_phillpotts.txt': 1, 'data/edgar_wallace.txt': -1, 'data/edith_lavell.txt': 1,\
 'data/edith_wharton.txt': 1, 'data/edmund_candler.txt': -1, 'data/edward_bennett.txt': -1,\
 'data/edward_booth.txt': -1, 'data/edward_dyson.txt': -1, 'data/edward_roe.txt': -1,\
 'data/edward_stratemeyer.txt': -1, 'data/edwarde_stratemeyer.txt': -1, 'data/edwin_brewster.txt': -1,\
 'data/edwin_bryant.txt': -1, 'data/eleanor_sidgwick.txt': 1, 'data/elia_peattie.txt': 1,\
 'data/elizabeth_haldane.txt': 1, 'data/ella_higginson.txt': 1, 'data/emerson_hough.txt': -1,\
 'data/emily-bronte.txt': 1, 'data/emily_hahn.txt': 1, 'data/emily_holt.txt': 1,\
 'data/emmuska_orczy.txt': 1, 'data/ethel_turner.txt': 1, 'data/evelyn_raymond.txt': 1,\
 'data/evelyn_scott.txt': 1, 'data/evelyn_sharp.txt': 1, 'data/evenlyn_everettgreen.txt': 1,\
 'data/everett_tomlinson.txt': -1, 'data/fanny_fern.txt': 1, 'data/flora_steel.txt': 1,\
 'data/florence_barclay.txt': 1, 'data/florence_coombe.txt': 1, 'data/ford_hueffer.txt': -1,\
 'data/frances_burnett.txt': 1, 'data/frank_herbert.txt': -1, 'data/frank_packard.txt': -1,\
 'data/frank_spearman.txt': -1, 'data/frank_tubbs.txt': -1, 'data/frederic_isham.txt': -1,\
 'data/frederica_turle.txt': 1, 'data/frederick_gordon.txt': -1, 'data/frederick_starr.txt': -1,\
 'data/george_ade.txt': 1, 'data/george_cable.txt': 1, 'data/george_fenn.txt': 1,\
 'data/george_grinnell.txt': -1, 'data/george_macdonald.txt': -1, 'data/geraldine_bonner.txt': 1,\
 'data/geraldine_mockler.txt': 1, 'data/gertrude_atherton.txt': 1, 'data/gertrude_morrison.txt': 1,\
 'data/gladys_allen.txt': 1, 'data/grace_hill.txt': 1, 'data/harold_bindloss.txt': -1,\
 'data/harold_goodwin.txt': -1, 'data/harriet_comstock.txt': 1, 'data/harriet_martineau.txt': 1,\
 'data/harriet_paine.txt': 1, 'data/harriet_stowe.txt': 1, 'data/harry_castlemon.txt': -1,\
 'data/helen_campbell.txt': 1, 'data/helen_keller.txt': 1, 'data/helen_martin.txt': 1,\
 'data/helen_rowland.txt': 1, 'data/henry_castlemon.txt': -1, 'data/henry_frith.txt': -1,\
 'data/henry_harland.txt': 1, 'data/henry_knibbs.txt': -1, 'data/henry_lewis.txt': -1,\
 'data/henry_merriman.txt': -1, 'data/henry_wace.txt': -1, 'data/henrydavid_thoreau.txt': -1,\
 'data/herbert_strang.txt': -1, 'data/heywood_broun.txt': -1, 'data/hezekiah_butterworth.txt': -1,\
 'data/horace_kephart.txt': -1, 'data/ian_may.txt': -1, 'data/ida_tarbell.txt': 1,\
 'data/isabel_mackay.txt': 1, 'data/isabella_bird.txt': 1, 'data/isabella_graham.txt': 1,\
 'data/jack_london.txt': -1, 'data/jackson_gregory.txt': -1, 'data/jacob_abbott.txt': -1,\
 'data/jacob_riis.txt': -1, 'data/james-joyce.txt': -1, 'data/james_bowker.txt': -1,\
 'data/james_connolly.txt': -1, 'data/james_schultz.txt': -1, 'data/jane-austen.txt': 1,\
 'data/jane_austin.txt': 1, 'data/jane_fryer.txt': 1, 'data/jean_webster.txt': 1,\
 'data/jeannie_gunn.txt': 1, 'data/jerome_jerome.txt': -1, 'data/joel_harris.txt': -1,\
 'data/john_bangs.txt': -1, 'data/john_gregory.txt': -1, 'data/john_mcgovern.txt': -1,\
 'data/john_ruskin.txt': -1, 'data/john_scott.txt': -1, 'data/john_tindall.txt': -1,\
 'data/jose_rizal.txt': -1, 'data/joseph-conrad.txt': -1, 'data/joseph_anderson.txt': -1,\
 'data/josephine_chase.txt': 1, 'data/josephine_culpeper.txt': 1, 'data/julia_dragoumis.txt': 1,\
 'data/julia_frankau.txt': 1, 'data/julia_magruder.txt': 1, 'data/julian_hawthorne.txt': 1,\
 'data/juliana_ewing.txt': 1, 'data/justus_forman.txt': -1, 'data/karl-brown.txt': -1,\
 'data/karl_harriman.txt': -1, 'data/kate_bosher.txt': 1, 'data/kate_chopin.txt': 1,\
 'data/kate_wiggin.txt': 1, 'data/laura_hope.txt': 1, 'data/laura_richards.txt': 1,\
 'data/laura_richardson.txt': 1, 'data/lawrence_burpee.txt': -1, 'data/leonard_merrick.txt': -1,\
 'data/lester_chadwick.txt': -1, 'data/lew_wallace.txt': -1, 'data/lewis-carroll.txt': -1,\
 'data/lillian_roy.txt': 1, 'data/louis_becke.txt': -1, 'data/louis_couperus.txt': -1,\
 'data/louis_freeman.txt': -1, 'data/louis_parker.txt': -1, 'data/louise_lamprey.txt': 1,\
 'data/lucile_lovell.txt': 1, 'data/luis_senarens.txt': -1, 'data/lydia_middleton.txt': 1,\
 'data/mabel_wright.txt': 1, 'data/madison_grant.txt': -1, 'data/malcolm_moreheart.txt': -1,\
 'data/margaret_deland.txt': 1, 'data/margaret_oliphant.txt': 1, 'data/margaret_pedler.txt': 1,\
 'data/margaret_penrose.txt': 1, 'data/margaret_robertson.txt': 1, 'data/margaret_sidney.txt': 1,\
 'data/margaret_vandercook.txt': 1, 'data/marguerite_merington.txt': 1, 'data/maria_parloa.txt': 1,\
 'data/marie_lowndes.txt': 1, 'data/marie_tarnowska.txt': 1, 'data/marietta_holley.txt': 1,\
 'data/marilyn_anderson.txt': 1, 'data/mark-twain.txt': -1, 'data/marshall_saunders.txt': -1,\
 'data/marth_ewell.txt': 1, 'data/martha_finley.txt': 1, 'data/mary-shelley.txt': 1,\
 'data/mary_austin.txt': 1, 'data/mary_code.txt': 1, 'data/mary_freeley.txt': 1,\
 'data/mary_holmes.txt': 1, 'data/mary_maclane.txt': 1, 'data/mary_mears.txt': 1,\
 'data/mary_molesworth.txt': 1, 'data/mary_waller.txt': 1, 'data/mary_ward.txt': 1,\
 'data/mary_waterbury.txt': 1, 'data/mary_webb.txt': 1, 'data/mary_weymss.txt': 1,\
 'data/maurice_maeterlinck.txt': -1, 'data/may_sinclair.txt': 1, 'data/mayne_reid.txt': -1,\
 'data/meg_gehrts.txt': 1, 'data/meredith_nicholson.txt': 1, 'data/mildred_wirt.txt': 1,\
 'data/millicent_fawcett.txt': 1, 'data/milo_oblinger.txt': -1, 'data/miriam_harris.txt': 1,\
 'data/nat_gould.txt': -1, 'data/nathaniel-hawthorne.txt': -1, 'data/oliver_optic.txt': -1,\
 'data/oscar-wilde.txt': -1, 'data/oscar_micheaux.txt': -1, 'data/ossip_schubin.txt': 1,\
 'data/paul_ernst.txt': -1, 'data/percy_brebner.txt': -1, 'data/percy_fitzhugh.txt': -1,\
 'data/phillip_schuler.txt': -1, 'data/phoebe_allen.txt': 1, 'data/ralph_barbour.txt': -1,\
 'data/richard_savage.txt': -1, 'data/robert_abernathy.txt': -1, 'data/robert_barr.txt': -1,\
 'data/robert_bowen.txt': -1, 'data/robert_chambers.txt': -1, 'data/robert_gilbert.txt': -1,\
 'data/robert_herrick.txt': -1, 'data/robert_howard.txt': -1, 'data/robert_knowles.txt': -1,\
 'data/robert_leckie.txt': -1, 'data/robert_leighton.txt': -1, 'data/robert_machray.txt': -1,\
 'data/robert_shaler.txt': -1, 'data/robert_stephens.txt': -1, 'data/rolf_boldrewood.txt': -1,\
 'data/rounsvelle_wildman.txt': -1, 'data/roy_rockwood.txt': -1, 'data/roy_snell.txt': -1,\
 'data/rudyard_kipling.txt': -1, 'data/samuel_butler.txt': -1, 'data/samuel_scoville.txt': -1,\
 'data/sara_pryor.txt': 1, 'data/sherwood_knight.txt': -1, 'data/sophie_may.txt': 1,\
 'data/stanley_matthews.txt': -1, 'data/stanley_weyman.txt': -1, 'data/stephen_cox.txt': -1,\
 'data/stephen_crane.txt': -1, 'data/stephen_mckenna.txt': -1, 'data/sterling_north.txt': -1,\
 'data/susan_coolidge.txt': -1, 'data/susanna_moodie.txt': 1, 'data/terence_hughes.txt': -1,\
 'data/thomas_burke.txt': -1, 'data/thornton_burgess.txt': -1, 'data/upton_sinclair.txt': -1,\
 'data/victor-hugo.txt': -1, 'data/violet_hunt.txt': 1, 'data/virginia-hughes.txt': 1,\
 'data/virginia-woolf.txt': 1, 'data/vivian_caulfeild.txt': 1, 'data/walt_whitman.txt': -1,\
 'data/washington_irving.txt': -1, 'data/wickham_hoffman.txt': -1, 'data/willa-cather.txt': 1,\
 'data/william_arnot.txt': -1, 'data/william_howells.txt': -1, 'data/william_lequeux.txt': -1,\
 'data/william_morris.txt': -1, 'data/william_shepard.txt': -1}

weights = learnPredictor(trainingLabels, None, extractWordFeatures, 200, 0.01)

#Printing out the weights in sorted order
for word, weight in sorted(weights.iteritems(), key=lambda word, weight: weight, word):
    print "%s: %s" % key, value

correctClassifications = 0
totalClassifications = len(trainingLabels)
#We'll change this later...decompose and stuff
for (file, y) in trainingLabels.iteritems():
	featureVector = extractWordFeatures(file)
	score = dotProduct(featureVector, weights)
	classification = -1 if score < 0 else 1
	if (classification == y):
		correctClassifications += 1
		print("Correctly classified %s as %d with score %f." % (file, classification, score))
	else:
		print("Incorrectly classified %s as %d with score %f." % (file, classification, score))
percentCorrect = 100 * float(correctClassifications) / totalClassifications
print("%d correct classifications out of %d (%f percent correct)." % (correctClassifications, totalClassifications, percentCorrect))


