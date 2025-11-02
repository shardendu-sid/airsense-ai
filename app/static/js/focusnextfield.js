$(function() {
  $(".mGrid >input").on('keyup', function(e) {
    if (e.which === 13) {
      $(this).next('input').focus();
    }
  });
});