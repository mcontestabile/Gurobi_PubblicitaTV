package it.unibs.gurobi.parte1;

import gurobi.*;

import static gurobi.GRB.IntAttr.VBasis;

/**
 * j = 1, 2, ..., K sono le fasce orarie televisive.
 * i = 1, . . . , M sono le emittenti televisive.
 * Cij costo €/minuto della pubblicità — asta combinatoria.
 * Pij spettatori/minuto di copertura.
 * τij limite di minuti (sicuramente un vincolo di ≤).
 * βi spesa massima consentita (sicuramente un vincolo di ≤).
 * Ω% budget minimo da spendere in j (sicuramente un vincolo di ≥).
 * S spettatori minimi che si vogliono raggiungere (sicuramente un vincolo di ≥).
 *
 * f.o. minimizzare lo scarto (differenza) tra le persone raggiunte nelle
 * prime K2 fasce orarie e le persone raggiunte nelle restanti. È la differenza
 * di due sommatorie.
 */
public class Elaborato_Gurobi {

    // Emittenti televisive.
    final static int m = 10;

    // Fasce orarie televisive.
    final static int k = 8;

    // Spettatori minimi che si vogliono raggiungere.
    final static int min_spectators = 82513;

    // Budget minimo da spendere in j
    final static int omega = 1;

    // Spesa massima consentita per emittente.
    final static int[] beta_i = {3406, 2517, 3440, 2577, 3431, 2629, 2548, 2518, 2678, 3286};

    // Costo €/minuto della pubblicità.
    final static int[][] cij = {{1264, 1205, 1389, 1267, 1071, 958, 1016, 1073},
            {967, 937, 981, 1128, 1301, 1141, 935, 1115},
            {1154, 1361, 1294, 1170, 1206, 1397, 1001, 1026},
            {1364, 1149, 997, 917, 1105, 1048, 1091, 1038},
            {1336, 1236, 1265, 1343, 988, 1293, 1057, 1164},
            {936, 1337, 961, 1175, 1340, 1070, 1115, 1312},
            {944, 1119, 1343, 902, 1048, 1372, 1286, 1113},
            {1090, 1336, 1167, 1145, 941, 922, 1030, 1096},
            {1231, 1253, 1254, 1006, 1343, 1169, 1280, 1118},
            {1129, 1021, 986, 977, 1343, 972, 930, 1303}};

    // Spettatori/minuto di copertura.
    final static int[][] pij = {{1999, 1603, 1371, 1927, 1668, 2867, 2152, 2342},
            {3173, 1620, 394, 2928, 1061, 1388, 1755, 1469},
            {1983, 3045, 1222, 2712, 1232, 2059, 1199, 483},
            {385, 1449, 2623, 614, 2039, 2479, 2801, 2619},
            {3013, 2384, 3365, 537, 1893, 1630, 1599, 2590},
            {3307, 1890, 2109, 1712, 1244, 2649, 1136, 2868},
            {3294, 872, 700, 542, 2552, 1839, 2570, 695},
            {2835, 762, 2762, 2933, 3487, 440, 3342, 679},
            {3209, 2242, 2547, 433, 2395, 958, 3330, 1358},
            {2026, 806, 3319, 1103, 1318, 451, 1546, 2957}};

    // Limite di minuti.
    final static int[][] tau_ij = {{2, 2, 2, 3, 2, 2, 2, 3},
            {2, 1, 3, 2, 2, 3, 2, 3},
            {2, 2, 1, 2, 2, 3, 1, 3},
            {2, 2, 1, 2, 1, 2, 2, 1},
            {2, 1, 2, 3, 1, 1, 1, 1},
            {3, 2, 1, 2, 2, 3, 3, 3},
            {2, 2, 2, 2, 2, 2, 2, 3},
            {2, 3, 2, 1, 2, 2, 3, 1},
            {2, 1, 3, 2, 1, 2, 2, 1},
            {2, 3, 2, 2, 2, 2, 1, 2}};

    static int counterS = 0; // contatore slacks;
    static int counterA = 0; // contatore ausiliarie.

    public static void main(String[] args) {

        try {
            GRBEnv env = new GRBEnv("Singolo_16.log");
            settingMyParameters(env);

            GRBModel model = new GRBModel(env);
            model.set(GRB.IntParam.OutputFlag, 0);

            // Matrice dei minuti effettivi acquistati.
            GRBVar[][] xij = settingMyVariables(model);

            // Funzione obiettivo linearizzata.
            GRBVar z = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "function");

            // Vincolo massimi minuti acquistabili.
            setConstraint1(model, xij);

            // Vincolo massimo euro investibile.
            setConstraint2(model, xij);

            // Vincolo minimo budget investibile (omega).
            setConstraint3(model, xij);

            // Vincolo spettatori.
            setConstraint4(model, xij);

            // Vincolo linearizzazione.
            setConstraintLinearization(model, xij, z);

            // Funzione obiettivo.
            setLossFunction(model, z);

            System.out.println("\n\n\nGuppo Singolo 16");
            System.out.println("Componenti: Contestabile");
            System.out.println("\n\n\n\n\n\nQUESITO I:");

            // Risolvo il modello lp.
            solve(model, xij);

            firstQuestion(model, xij);
            secondQuestion(model, xij);

            // Salvo quello che ho ottenuto col simplesso.
            GRBVar[] simplex = new GRBVar[model.getVars().length];

            thirdQuestion(simplex, env);

            model.dispose();
            env.dispose();

        } catch (GRBException e) {
            e.printStackTrace();
        }
    }

    private static void settingMyParameters(GRBEnv env) throws GRBException {
        env.set(GRB.IntParam.Method, 0);
        env.set(GRB.IntParam.Presolve, 0);
    }

    /**
     * Imposto i parametri del modello del problema di ottimizzazione.
     *
     * @param model modello del problema di ottimizzazione.
     * @return xij                matrice dei minuti effettivi acquistati.
     * @throws GRBException
     */
    private static GRBVar[][] settingMyVariables(GRBModel model) throws GRBException {
        GRBVar[][] xij = new GRBVar[m][k];

        for (int p = 0; p < m; p++)
            for (int q = 0; q < k; q++)
                xij[p][q] = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "xij_" + p + "," + q);

        return xij;
    }


    /**
     * Costruzione della funzione obiettivo, la cui struttura è
     * * w = sommatoria1 - sommatoria2
     * * z = modulo di w [ questa variabile viene linearizzata ]
     *
     * @param model    modello del problema di ottimizzazione.
     * @param function variabile rappresentante la funzione obiettivo.
     * @throws GRBException
     */
    private static void setLossFunction(GRBModel model, GRBVar function) throws GRBException {
        // z = max { w, - w }
        GRBLinExpr z = new GRBLinExpr();

        z.addTerm(1, function);

        model.setObjective(z);
        model.set(GRB.IntAttr.ModelSense, GRB.MINIMIZE);
    }

    /**
     * xij deve essere ≤ di tau_ij
     * linearizzato, significa che xij + s = tau_ij
     *
     * @param model modello del problema di ottimizzazione.
     * @param xij   matrice dei minuti effettivi acquistati.
     * @throws GRBException
     */
    private static void setConstraint1(GRBModel model, GRBVar[][] xij) throws GRBException {
        // Parto dalle righe della matrice xij
        for (int p = 0; p < m; p++)
            for (int q = 0; q < k; q++) {
                GRBLinExpr constraintTauij = new GRBLinExpr();
                constraintTauij.addTerm(1, xij[p][q]);
                GRBVar slack_tau_ij = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "slack_tau_" + p + "," + q);
                constraintTauij.addTerm(1, slack_tau_ij);

                model.addConstr(constraintTauij, GRB.EQUAL, tau_ij[p][q], "vincolo_tempo_ij_" + p + "," + q);
            }

    }

    /**
     * Spesa massima per emittente.
     *
     * @param model modello del problema di ottimizzazione.
     * @param xij   matrice dei minuti effettivi acquistati.
     * @throws GRBException
     */
    private static void setConstraint2(GRBModel model, GRBVar[][] xij) throws GRBException {
        for (int p = 0; p < m; p++) {
            GRBLinExpr beta_constraint = new GRBLinExpr();

            for (int q = 0; q < k; q++)
                beta_constraint.addTerm(cij[p][q], xij[p][q]);

            GRBVar beta_slack = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "slack_beta_" + p);
            beta_constraint.addTerm(1, beta_slack);

            model.addConstr(beta_constraint, GRB.EQUAL, beta_i[p], "vincolo_beta_i_" + p);
        }
    }

    /**
     * Budget minimo fascia.
     *
     * @param model modello del problema di ottimizzazione.
     * @param xij   matrice dei minuti effettivi acquistati.
     * @throws GRBException
     */
    private static void setConstraint3(GRBModel model, GRBVar[][] xij) throws GRBException {
        for (int p = 0; p < k; p++) {
            GRBLinExpr omega_constraint = new GRBLinExpr();

            for (int q = 0; q < m; q++)
                omega_constraint.addTerm(cij[q][p], xij[q][p]);


            GRBVar omega_slack = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "omega_slack" + p);
            omega_constraint.addTerm(-1, omega_slack);

            model.addConstr(omega_constraint, GRB.EQUAL, percentageBudget(), "vincolo_fascia_" + p);
        }
    }

    /**
     * Copertura minima giornaliera.
     *
     * @param model modello del problema di ottimizzazione.
     * @param xij   matrice dei minuti effettivi acquistati.
     * @throws GRBException
     */
    private static void setConstraint4(GRBModel model, GRBVar[][] xij) throws GRBException {
        GRBLinExpr constraintSpectators = new GRBLinExpr();

        for (int p = 0; p < m; p++)
            for (int q = 0; q < k; q++)
                constraintSpectators.addTerm(pij[p][q], xij[p][q]);

        GRBVar slack_spectators = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "slack_spettatori");
        constraintSpectators.addTerm(-1, slack_spectators);

        model.addConstr(constraintSpectators, GRB.EQUAL, min_spectators, "vincolo_spettatori");
    }

    /**
     * Vincolo 1 della funzione obiettivo linearizzata, ovvero quando
     * z = max { w, - w } è max quando si ottiene, per w = a
     * e -w = b, che b ≥ a e, quindi, il massimo è b.
     * <p>
     * Ora devo fare che w ≥ - function.
     *
     * @param model    modello del problema di ottimizzazione.
     * @param function funzione obiettivo da linearizzare.
     * @param xij      variabili del problema.
     * @throws GRBException
     */
    private static void setConstraintLinearization(GRBModel model, GRBVar[][] xij, GRBVar function) throws GRBException {
        GRBLinExpr w1 = new GRBLinExpr();
        GRBLinExpr w2 = new GRBLinExpr();

        GRBVar slack1 = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "slack_w1");
        GRBVar slack2 = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "slack_w2");
        for (int p = 0; p < m; p++)
            for (int q = 0; q < k; q++) {
                int plusOrMinus = (q < (k/2) ? 1 : -1);
                w1.addTerm(plusOrMinus * pij[p][q], xij[p][q]);
                w2.addTerm((-1 * plusOrMinus) * pij[p][q], xij[p][q]);
            }

        w1.addTerm(-1, function);
        w2.addTerm(1, function);

        w1.addTerm(1, slack1);
        w2.addTerm(-1, slack2);

        model.addConstr(w1, GRB.EQUAL, 0, "linearizzazione_1");
        model.addConstr(w2, GRB.EQUAL, 0, "linearizzazione_2");
    }


    /**
     * xij deve essere ≤ di tau_ij
     * linearizzato, significa che xij + s = tau_ij
     *
     * @param model modello del problema di ottimizzazione.
     * @param xij   matrice dei minuti effettivi acquistati.
     * @param aux
     * @throws GRBException
     */
    private static void setConstraint1_2(GRBModel model, GRBVar[][] xij, GRBVar[] aux, GRBVar[] slacks) throws GRBException {
        // Parto dalle righe della matrice xij
        for (int p = 0; p < m; p++)
            for (int q = 0; q < k; q++) {
                GRBLinExpr constraintTauij_2 = new GRBLinExpr();
                constraintTauij_2.addTerm(1, xij[p][q]);
                GRBVar slack_tau_ij_2 = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "slack_tau_(2)_" + p + "," + q);
                constraintTauij_2.addTerm(1, slack_tau_ij_2);
                constraintTauij_2.addTerm(1, aux[counterA]);
                counterA++;

                model.addConstr(constraintTauij_2, GRB.EQUAL, tau_ij[p][q], "vincolo_tempo_ij_(2)_" + p + "," + q);
            }

    }

    /**
     * Spesa massima per emittente.
     *
     * @param model modello del problema di ottimizzazione.
     * @param xij   matrice dei minuti effettivi acquistati.
     * @param aux
     * @throws GRBException
     */
    private static void setConstraint2_2(GRBModel model, GRBVar[][] xij, GRBVar[] aux, GRBVar[] slacks) throws GRBException {
        for (int p = 0; p < m; p++) {
            GRBLinExpr beta_constraint_2 = new GRBLinExpr();

            for (int q = 0; q < k; q++)
                beta_constraint_2.addTerm(cij[p][q], xij[p][q]);

            GRBVar beta_slack_2 = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "slack_beta_(2)_" + p);
            beta_constraint_2.addTerm(1, beta_slack_2);
            beta_constraint_2.addTerm(1, aux[counterS]);

            counterA++;

            model.addConstr(beta_constraint_2, GRB.EQUAL, beta_i[p], "vincolo_beta_(2)_i_" + p);
        }
    }

    /**
     * Budget minimo fascia.
     *
     * @param model modello del problema di ottimizzazione.
     * @param xij   matrice dei minuti effettivi acquistati.
     * @param aux
     * @throws GRBException
     */
    private static void setConstraint3_2(GRBModel model, GRBVar[][] xij, GRBVar[] aux, GRBVar[] slacks) throws GRBException {
        for (int q = 0; q < k; q++) {
            GRBLinExpr omega_constraint_2 = new GRBLinExpr();

            for (int p = 0; p < m; p++)
                omega_constraint_2.addTerm(cij[p][q], xij[p][q]);


            GRBVar omega_slack_2 = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "omega_(2)_slack" + q);
            omega_constraint_2.addTerm(-1, omega_slack_2);
            omega_constraint_2.addTerm(1, aux[counterS]);

            counterA++;

            model.addConstr(omega_constraint_2, GRB.EQUAL, percentageBudget(), "vincolo_(2)_fascia_" + q);
        }
    }

    /**
     * Copertura minima giornaliera.
     *
     * @param model modello del problema di ottimizzazione.
     * @param xij   matrice dei minuti effettivi acquistati.
     * @param aux
     * @throws GRBException
     */
    private static void setConstraint4_2(GRBModel model, GRBVar[][] xij, GRBVar[] aux, GRBVar[] slacks) throws GRBException {
        GRBLinExpr constraintSpectators_2 = new GRBLinExpr();

        for (int p = 0; p < m; p++)
            for (int q = 0; q < k; q++)
                constraintSpectators_2.addTerm(pij[p][q], xij[p][q]);

        GRBVar slack_spectators_2 = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "slack_(2)_spettatori");
        constraintSpectators_2.addTerm(-1, slack_spectators_2);
        constraintSpectators_2.addTerm(1, aux[counterS]);

        counterA++;

        model.addConstr(constraintSpectators_2, GRB.EQUAL, min_spectators, "vincolo_(2)_spettatori");
    }


    /**
     * Costruzione della funzione obiettivo, la cui struttura è
     * * w = sommatoria1 - sommatoria2
     * * z = modulo di w [ questa variabile viene linearizzata ]
     *
     * @param model    modello del problema di ottimizzazione.
     * @param fo_aux variabile rappresentante la funzione obiettivo.
     * @throws GRBException
     */
    private static void setLossFunction_2(GRBModel model, GRBLinExpr fo_aux, GRBVar[] aux) throws GRBException {
        for (GRBVar g : aux)
            fo_aux.addTerm(1, g);

        model.setObjective(fo_aux);
        model.set(GRB.IntAttr.ModelSense, GRB.MINIMIZE);
    }


    /**
     * Percentuale del budget.
     *
     * @return la percentuale totale investibile.
     */
    private static double percentageBudget() {
        int sum = 0;
        for (int k : beta_i) sum += k;
        return (sum / 100) * omega;
    }

    private static double purchasedTime(GRBVar[][] xij) throws GRBException {
        int purchasedTime = 0;
        for (int p = 0; p < m; p++)
            for (int q = 0; q < k; q++)
                purchasedTime += xij[p][q].get(GRB.DoubleAttr.X);

        return purchasedTime;
    }

    private static double timeBoughtCost(GRBVar[][] xij) throws GRBException {
        int timeBoughtCost = 0;
        for (int p = 0; p < m; p++)
            for (int q = 0; q < k; q++)
                timeBoughtCost += (xij[p][q].get(GRB.DoubleAttr.X) * cij[p][q]);

        return timeBoughtCost;
    }

    /**
     * Copertura giornaliera ottenuta.
     * <p>
     * doppia sommatoria di pij * xij
     *
     * @param xij matrice dei minuti effettivi acquistati.
     * @throws GRBException
     */
    private static double obtainedShare(GRBVar[][] xij) throws GRBException {
        double obtainedShare = 0;
        for (int p = 0; p < m; p++)
            for (int q = 0; q < k; q++)
                obtainedShare += (xij[p][q].get(GRB.DoubleAttr.X) * pij[p][q]);

        return Math.round(obtainedShare);
    }

    /**
     * Metodo per trovare il budget rimasto.
     * @param xij
     * @return
     * @throws GRBException
     */
    private static double budgetLeft(GRBVar[][] xij) throws GRBException {
        double budget = 0;
        for (int k : beta_i) budget += k;
        return (budget - timeBoughtCost(xij));
    }

    private static double round(double value) {
        int exp = 4;
        int base = 10;
        return (Math.round(value * Math.pow(base, exp)) / Math.pow(base, exp));
    }

    private static void firstQuestion(GRBModel model, GRBVar[][] xij) throws GRBException {
        double budgetLeft = budgetLeft(xij);
        double obtainedShare = obtainedShare(xij);
        double purchasedTime = purchasedTime(xij);


        System.out.println("Funzione obiettivo = " + round(model.get(GRB.DoubleAttr.ObjVal)));
        System.out.println("Budget rimasto = " + budgetLeft);
        System.out.println("Share ottenuto = " + obtainedShare);
        System.out.println("Minuti acquistati = " + purchasedTime);
        System.out.println("Soluzione di base e fuori base all'ottimo:");


        for (GRBVar c : model.getVars())
            if (!c.get(GRB.StringAttr.VarName).equals("function"))
                System.out.println(c.get(GRB.StringAttr.VarName) + " = " + c.get(GRB.DoubleAttr.X));
        model.write("modello.lp");
    }

    private static void secondQuestion(GRBModel model, GRBVar[][] xij) throws GRBException {
        System.out.println("\n\n\n\n\n\nQUESITO II:");
        int basis = 0;

        System.out.println("\nVariabili in base:");
        for (GRBVar c : model.getVars())
            if (c.get(GRB.IntAttr.VBasis) == 0) {
                System.out.println(c.get(GRB.StringAttr.VarName) + " = 1");
                basis++;
            }


        System.out.println("\nVariabili fuori base:");
        for (GRBVar c : model.getVars())
            if (c.get(GRB.IntAttr.VBasis) != 0)
                System.out.println(c.get(GRB.StringAttr.VarName) + " = 0");


        // Qui verifico l'intersezione, se c'è, allora è un vertice.
        System.out.println("\nVincoli vertice ottimo:");
        for (int p = (m * k) + 1; p < model.getVars().length; p++)
            if (model.getVars()[p].get(GRB.IntAttr.VBasis) == 0)
                System.out.println(model.getVars()[p].get(GRB.StringAttr.VarName) + " = " + model.getVars()[p].get(GRB.IntAttr.VBasis));


        System.out.println("\nCoefficienti di costo ridotto:");
        // xij.
        for (GRBVar var : model.getVars())
            System.out.println("RC di " + var.get(GRB.StringAttr.VarName) + " = " + var.get(GRB.DoubleAttr.RC));

        // Le slack.
        for (int p = 0; p < m; p++)
            for (int q = 0; q < k; q++)
                System.out.println("RC di " + xij[p][q].get(GRB.StringAttr.VarName) + " = " + round(xij[p][q].get(GRB.DoubleAttr.RC)));


        /*
         *                  ANNOTAZIONE PER RISOLVERE IL QUESITO SUCCESSIVO
         *
         * Teorema: Sia dato un problema di programmazione lineare e una sua base ammissibile B.
         *          Se tutti i costi ridotti rispetto alla base B sono non negativi (≥ 0), allora
         *          la soluzione di base associata a B è ottima.
         */


        /*
         * Stabilisco se l'ottimo è multiplo.
         * Succede se RC > vincoli.
         */
        String multipleOptimum = "No.";
        int ccr_0 = 0;
        for (GRBVar var : model.getVars())
            if (var.get(GRB.DoubleAttr.RC) == 0)
                ccr_0++;
        if (ccr_0 > 101)
            multipleOptimum = "Sì.";

        // Controllo se è degenere.
        String isDegeneres = "No.";
        for (GRBVar var : model.getVars())
            /*
             * La soluzione ottima è detta degenere se anche
             * solo una variabile è in base + ha valore pari a zero.
             */
            if (var.get(VBasis) == 0 && var.get(GRB.DoubleAttr.X) == 0) {
                isDegeneres = "Sì.";
                break;
            }


        // Stabilisco cosa stampare in base ai risultati ottenuti.
        System.out.println("Soluzione ottima multipla: " + multipleOptimum);
        System.out.println("Soluzione ottima degenere: " + isDegeneres);

        int n = model.getVars().length;
        System.out.println("n = " + n);     // 182, ossia 80 xij + 1 della f.o. + le restanti 101 delle slack.
        System.out.println("m = " + basis); // Variabili in base.

        int slacksSize = model.getVars().length - ((m * k) + 1);
        GRBVar[] slacks = new GRBVar[slacksSize]; // 182 - (8 * 10 + 1).
        for (int i = 0; i < slacksSize; i++)
            slacks[i] = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "sla " + i);


        int slacksCount = 0;
        for (int v = (m * k) + 1; v < model.getVars().length; v++) {
            slacks[slacksCount] = model.getVars()[v];
            slacksCount++;
        }
        System.out.println("slacksCount = " + slacksCount); // Variabili in base.
    }

    private static GRBVar[] setSlackOrSurplus(GRBModel model) throws GRBException {
        GRBVar[] s = new GRBVar[m * k + m + k + 1];

        for (int i = 0; i < (m * k + m + k + 1); i++)
            s[i] = model.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "s_" + i);

        return s;
    }

    private static void thirdQuestion(GRBVar[] simplex, GRBEnv env) throws GRBException {
        // TODO l'ausiliaria ritorna come valore OBJ zero e non permette poi di fare la combinazione convessa!!!
        /*
         * I PROCEDURA per trovare la soluzione ammissibile non ottima.
         * Ciò che questa parte di codice svolge è la ricerca di una soluzione
         * ammissibile tramite il problema ausiliario.
         */
        GRBModel model2 = new GRBModel(env);
        //model2.set(GRB.IntParam.OutputFlag, 0); mi dà 2, il modello matematico è corretto.

        // Ausiliaria.
        GRBVar[] auxiliary = new GRBVar[101]; // 2 + 10 * 8 + 10 + 8 + 1
        for (int i = 0; i < 101; i++)
            auxiliary[i] = model2.addVar(0.0, GRB.INFINITY, 0.0, GRB.CONTINUOUS, "ausiliaria_" + i);

        // III Domanda.
        System.out.printf("QUESITO III:\n");

        // Matrice dei minuti effettivi acquistati.
        GRBVar[][] xij = settingMyVariables(model2);

        // Nuova funzione obiettivo.
        GRBLinExpr fo_aux = new GRBLinExpr();

        // Setto le slack
        GRBVar[] slacks = setSlackOrSurplus(model2);

        GRBVar z = model2.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "function");

        // Aggiunta dei vari vincoli.
        setConstraint1_2(model2, xij, auxiliary, slacks);
        setConstraint2_2(model2, xij, auxiliary, slacks);
        setConstraint3_2(model2, xij, auxiliary, slacks);
        setConstraint4_2(model2, xij, auxiliary, slacks);
        setLossFunction_2(model2, fo_aux, auxiliary);

        // Risolvo il problema ausiliario.
        solve(model2, xij);

        // Mi salvo le variabili del problema ausiliario.
        int tmp = 0;
        String firstNonOptimal = "I soluzione ammissibile non ottima:";
        GRBVar[] auxiliaryProblemVars = new GRBVar[k * m + 101];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < k; j++) {
                auxiliaryProblemVars[tmp] = xij[i][j];
                firstNonOptimal += "\n" + xij[i][j].get(GRB.StringAttr.VarName) + " = " + round(xij[i][j].get(GRB.DoubleAttr.X));
                tmp++;
            }
        }

        for (int i = 0; i < slacks.length; i++) {
            auxiliaryProblemVars[m * k + i] = slacks[i];
            firstNonOptimal += "\n" + slacks[i].get(GRB.StringAttr.VarName) + " = " + round(slacks[i].get(GRB.DoubleAttr.X)) + "\n";
        }

        String secondNonOptimal = "II soluzione ammissibile non ottima:";
        for (GRBVar var : model2.getVars()) {
            secondNonOptimal += "\n" + var.get(GRB.StringAttr.VarName) + " = " + round(var.get(GRB.DoubleAttr.X));
        }

        /*
         * II PROCEDURA: Eseguo la combinazione convessa fra la soluzione
         * ricavata dalla prima fase e quella ottima. Come abbiamo visto a
         * lezione, è più semplice ricavarsi il punto medio tra loro due.
         */
        double lambda = 0.5; // Per ottenere il punto medio, lambda deve necessariamente valere un mezzo.
        double[] secondSolution = new double[tmp];

        for (int i = 0; i < tmp; i++) {
            secondSolution[i] = lambda * auxiliaryProblemVars[i].get(GRB.DoubleAttr.X) + (1-lambda) * simplex[i].get(GRB.DoubleAttr.X);
        }

        String thirdNonOptimal = "III soluzione ammissibile non ottima\n";
        for (int i = 0; i < secondSolution.length; i++)
            thirdNonOptimal += auxiliaryProblemVars[i].get(GRB.StringAttr.VarName) + " = " + round(secondSolution[i]) + "\n";

        model2.dispose();
        env.dispose();

        /*
         * III Procedura: è sempre possibile aggiungere e/o togliere vincoli, quindi decido
         * di crearmi un nuovo vincolo (per non crearmi problemi togliendone) sulla mia
         * incognita (x) e mi ricavo una delle possibili soluzioni ammissibili.
         *
         * Scelgo l'inizio della matrice, xij[0][0]
         */
        GRBLinExpr ex = new GRBLinExpr();
        GRBModel model = new GRBModel(env);
        ex.addTerm(1, xij[0][0]);

        model.addConstr(ex, GRB.EQUAL, 1, "vincolo_origine");
        // Vincolo massimi minuti acquistabili.
        setConstraint1(model, xij);

        // Vincolo massimo euro investibile.
        setConstraint2(model, xij);

        // Vincolo minimo budget investibile (omega).
        setConstraint3(model, xij);

        // Vincolo spettatori.
        setConstraint4(model, xij);

        // Vincolo linearizzazione.
        setConstraintLinearization(model, xij, z);

        GRBVar function = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "function");
        // Funzione obiettivo.
        setLossFunction(model,function);
        model.optimize();
        System.out.printf("III soluzione ammissibile non ottima:");
        //stampa sol ammissibile 1
        for(GRBVar g : model.getVars()) {
            if(!g.get(GRB.StringAttr.VarName).equals("function"))
                System.out.println(g.get(GRB.StringAttr.VarName) + " = " + g.get(GRB.DoubleAttr.X));
        }
    }

    private static void solve(GRBModel model, GRBVar[][] xij) throws GRBException {

        model.optimize();


        int status = model.get(GRB.IntAttr.Status); // 2 soluzione ottima trovata.
        //System.out.println("status: " + status);
    }
}

