$(document).ready(function() {
      // Inizializza i datepicker
      $("#start-date").datepicker({ dateFormat: 'dd/mm/yy' });
      $("#end-date").datepicker({ dateFormat: 'dd/mm/yy' });

      // Aggiungi un listener al click di un bottone o a un evento di tua scelta
      $("#generate-url-button").on("click", function() {
        // Ottieni i valori delle date
        let startDate = $("#start-date").datepicker("getDate");
        let endDate = $("#end-date").datepicker("getDate");

        // Formatta le date come stringhe nel formato "YYYY-MM-DD"
        let formattedStartDate = $.datepicker.formatDate("yy-mm-dd", startDate);
        let formattedEndDate = $.datepicker.formatDate("yy-mm-dd", endDate);
        console.log(formattedStartDate)
        console.log(formattedEndDate)

        // Crea l'URL con le date formattate
        window.location = "http://localhost:8080/make_planning/start=" + formattedStartDate + "&end=" + formattedEndDate;
      });
    });